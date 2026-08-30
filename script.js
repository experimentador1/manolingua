import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const demosSection = document.getElementById("demos");
const letterDiv = document.getElementById("letterDiv");
const letterText = document.getElementById("letterText");
const imagen = document.getElementById("imagenModal");
const imagenLetter = document.getElementById("imageLetter");
const VIDEO = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const STATUS = document.getElementById("status");
const PREDICT = document.getElementById("predict");
const videoStage = document.getElementById("videoStage");
const cameraHint = document.getElementById("cameraHint");

const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const MOBILENET_URL = "./resources/mobilenet/model.json";
// Confianza mínima (%) para dar la seña por correcta.
const SUCCESS_THRESHOLD = 80;

let handLandmarker = undefined;
let webcamRunning = false;
let modelCustom = null;
let predict = false;
let selectedLetter = null;
let mobilenet = undefined;
let lastVideoTime = -1;
let results = undefined;
let classNames = ["Palma", "?"];
let modelsReady = false;

function setStatus(message) {
  STATUS.textContent = message || "";
}

function waitForGlobal(name, timeoutMs = 20000) {
  return new Promise((resolve, reject) => {
    if (window[name]) {
      resolve(window[name]);
      return;
    }
    const started = Date.now();
    const timer = setInterval(() => {
      if (window[name]) {
        clearInterval(timer);
        resolve(window[name]);
      } else if (Date.now() - started > timeoutMs) {
        clearInterval(timer);
        reject(new Error(`${name} no se cargó a tiempo.`));
      }
    }, 50);
  });
}

async function createHandLandmarker() {
  setStatus("Cargando detector de manos...");
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );

  const modelAssetPath =
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

  const options = { runningMode: "VIDEO", numHands: 1 };

  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath, delegate: "GPU" },
      ...options,
    });
  } catch (gpuError) {
    console.warn("GPU no disponible, usando CPU:", gpuError);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath, delegate: "CPU" },
      ...options,
    });
  }

  demosSection.classList.remove("invisible");
}

async function loadMobileNetFeatureModel() {
  const tf = await waitForGlobal("tf");
  setStatus("Cargando MobileNet...");
  mobilenet = await tf.loadGraphModel(MOBILENET_URL);
  tf.tidy(() => {
    const answer = mobilenet.predict(
      tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3])
    );
    console.log("MobileNet listo:", answer.shape);
  });
}

async function bootstrap() {
  try {
    await Promise.all([
      createHandLandmarker(),
      loadMobileNetFeatureModel(),
      waitForGlobal("drawConnectors"),
      waitForGlobal("HAND_CONNECTIONS"),
    ]);
    modelsReady = true;
    PREDICT.textContent = "Sin letra seleccionada";
    PREDICT.style.display = "block";
    await enableCam();
  } catch (error) {
    console.error(error);
    demosSection.classList.remove("invisible");
    setStatus(
      "Error al cargar recursos. Revisa tu conexión a internet y recarga la página."
    );
  }
}

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

async function enableCam() {
  if (!modelsReady || !handLandmarker || webcamRunning) {
    return;
  }
  if (!hasGetUserMedia()) {
    setStatus("Tu navegador no soporta acceso a la webcam.");
    return;
  }

  try {
    setStatus("Abriendo cámara…");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
    VIDEO.srcObject = stream;
    VIDEO.muted = true;
    await VIDEO.play();

    webcamRunning = true;
    predict = true;
    PREDICT.style.display = "block";
    videoStage.classList.add("active");
    setStatus(
      selectedLetter
        ? `Reconociendo letra ${selectedLetter}. Muestra tu mano frente a la cámara.`
        : "Cámara lista. Elige una letra del menú (⋮) para empezar."
    );

    startRenderLoop();
    predictLoop();
  } catch (error) {
    console.error(error);
    if (cameraHint) {
      cameraHint.textContent =
        "No se pudo abrir la cámara. Permite el acceso y recarga la página.";
    }
    setStatus(
      "No se pudo acceder a la cámara. Revisa permisos del navegador (localhost/HTTPS)."
    );
  }
}

function startRenderLoop() {
  // HAVE_CURRENT_DATA o superior: ya hay frame utilizable.
  if (VIDEO.readyState >= 2) {
    predictWebcam();
  } else {
    VIDEO.addEventListener("loadeddata", predictWebcam, { once: true });
  }
}

// drawing_utils descarta puntos con visibility <= 0.5 y tasks-vision los
// entrega con visibility 0, así que se omite el campo para poder dibujarlos.
function stripVisibility(landmarks) {
  return landmarks.map((point) => ({ x: point.x, y: point.y, z: point.z }));
}

// El canvas contiene únicamente el esqueleto de la mano: es la entrada
// exacta con la que se entrenaron los modelos por letra.
function predictWebcam() {
  if (!webcamRunning || !handLandmarker) {
    return;
  }

  const width = VIDEO.videoWidth || 640;
  const height = VIDEO.videoHeight || 480;

  if (canvasElement.width !== width || canvasElement.height !== height) {
    canvasElement.width = width;
    canvasElement.height = height;
  }

  const startTimeMs = performance.now();
  if (lastVideoTime !== VIDEO.currentTime) {
    lastVideoTime = VIDEO.currentTime;
    try {
      results = handLandmarker.detectForVideo(VIDEO, startTimeMs);
    } catch (error) {
      console.error("Error en detección de mano:", error);
    }
  }

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  if (results && results.landmarks) {
    for (const rawLandmarks of results.landmarks) {
      const landmarks = stripVisibility(rawLandmarks);
      window.drawConnectors(canvasCtx, landmarks, window.HAND_CONNECTIONS, {
        color: "#00FF00",
        lineWidth: 2,
      });
      window.drawLandmarks(canvasCtx, landmarks, {
        color: "#FF0000",
        lineWidth: 1,
      });
    }
  }
  canvasCtx.restore();

  window.requestAnimationFrame(predictWebcam);
}

function hasHand() {
  return !!(results && results.landmarks && results.landmarks.length > 0);
}

function calculateFeaturesOnCurrentFrame(tf) {
  return tf.tidy(() => {
    if (!canvasElement.width || !canvasElement.height) {
      return null;
    }
    const videoFrameAsTensor = tf.browser.fromPixels(canvasElement);
    const resizedTensorFrame = tf.image.resizeBilinear(
      videoFrameAsTensor,
      [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
      true
    );
    const normalizedTensorFrame = resizedTensorFrame.div(255);
    return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
  });
}

function predictLoop() {
  if (!predict) {
    return;
  }

  const tf = window.tf;
  if (!tf || !mobilenet || !modelCustom || !webcamRunning) {
    window.requestAnimationFrame(predictLoop);
    return;
  }

  try {
    if (!hasHand()) {
      PREDICT.textContent = "Muestra tu mano frente a la cámara";
      letterText.style.color = "white";
    } else {
      tf.tidy(() => {
        const imageFeatures = calculateFeaturesOnCurrentFrame(tf);
        if (!imageFeatures) {
          return;
        }
        const prediction = modelCustom.predict(imageFeatures.expandDims()).squeeze();
        const highestIndex = prediction.argMax().arraySync();
        const predictionArray = prediction.arraySync();
        const predictRate = Math.floor(predictionArray[highestIndex] * 100);

        if (classNames[highestIndex] === "Palma") {
          PREDICT.textContent = "SIGUE INTENTANDO";
          letterText.style.color = "white";
        } else if (predictRate >= SUCCESS_THRESHOLD) {
          PREDICT.textContent = `LO TIENES (${predictRate}%)`;
          letterText.style.color = "rgb(245, 116, 116)";
        } else {
          PREDICT.textContent = `CASI LO TIENES (${predictRate}%)`;
          letterText.style.color = "white";
        }
      });
    }
  } catch (error) {
    console.error("Error en predicción:", error);
  }

  window.requestAnimationFrame(predictLoop);
}

function resolveLetterAssets(option) {
  const normalized = option.trim().normalize("NFC").toUpperCase();
  // En disco la Ñ está guardada como N + tilde combinante (NFD).
  if (normalized === "Ñ") {
    const nTilde = "N\u0303";
    return { letter: "Ñ", folder: nTilde, screenshot: `${nTilde}.png` };
  }
  return {
    letter: normalized,
    folder: normalized,
    screenshot: `${normalized}.png`,
  };
}

async function loadLetter(option) {
  const tf = await waitForGlobal("tf");
  const { letter, folder, screenshot } = resolveLetterAssets(option);

  selectedLetter = letter;
  letterText.textContent = letter;
  letterDiv.style.visibility = "visible";

  const screenshotPath = `resources/screenshots/${screenshot}`;
  imagen.src = screenshotPath;
  imagenLetter.src = screenshotPath;
  imagenLetter.alt = `Referencia letra ${letter}`;

  setStatus(`Cargando modelo de la letra ${letter}...`);
  PREDICT.textContent = `Letra ${letter} seleccionada`;
  PREDICT.style.display = "block";

  try {
    modelCustom = await tf.loadLayersModel(
      `./resources/models/${folder}/modelCustom.json`
    );
    classNames = ["Palma", letter];
    setStatus(`Reconociendo letra ${letter}. Muestra tu mano frente a la cámara.`);

    if (webcamRunning) {
      predict = true;
      predictLoop();
    } else if (modelsReady) {
      await enableCam();
    }
  } catch (error) {
    console.error(error);
    modelCustom = null;
    setStatus(`No se pudo cargar el modelo de la letra ${letter}.`);
  }
}

document.querySelectorAll(".list-group-item").forEach((enlace) => {
  enlace.addEventListener("click", (event) => {
    event.preventDefault();
    loadLetter(enlace.textContent.trim());
  });
});

bootstrap();
