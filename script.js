// Load the CLIP tools (Transformers.js) from a CDN
import {
  AutoTokenizer,
  CLIPTextModelWithProjection,
  AutoProcessor,
  CLIPVisionModelWithProjection,
  RawImage
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0";

var MODEL_ID = "Xenova/clip-vit-base-patch32";

// parts of the page
var photoInput = document.getElementById("photoInput");
var folderButton = document.getElementById("folderButton");
var progressBar = document.getElementById("progressBar");
var status = document.getElementById("status");
var descriptionInput = document.getElementById("descriptionInput");
var searchButton = document.getElementById("searchButton");
var results = document.getElementById("results");

// the AI models (filled in when they finish loading)
var tokenizer;
var textModel;
var processor;
var visionModel;

// every indexed photo: { url: "...", embedding: [...] }
var indexedPhotos = [];

// ---------- 1. load the models ----------
async function loadModels() {
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  textModel = await CLIPTextModelWithProjection.from_pretrained(MODEL_ID);
  processor = await AutoProcessor.from_pretrained(MODEL_ID);
  visionModel = await CLIPVisionModelWithProjection.from_pretrained(MODEL_ID);
  status.textContent = "Ready! Choose your photos to begin.";
}

// ---------- 2. turn photos into embeddings (lists of numbers) ----------
async function indexPhotos(files) {
  indexedPhotos = [];
  results.innerHTML = "";
  searchButton.disabled = true;
  progressBar.style.display = "block";
  progressBar.max = files.length;
  progressBar.value = 0;

  for (var i = 0; i < files.length; i++) {
    status.textContent = "Indexing photo " + (i + 1) + " of " + files.length + "...";

    try {
      var url = URL.createObjectURL(files[i]);
      var image = await RawImage.read(url);
      var imageInputs = await processor(image);
      var output = await visionModel(imageInputs);
      var embedding = output.image_embeds.normalize(2, -1).data;

      indexedPhotos.push({ url: url, embedding: embedding });
    } catch (error) {
      // some files (like HEIC photos) can't be read, so skip them
      console.log("Skipped a photo:", files[i].name, error);
    }

    progressBar.value = i + 1;
  }

  progressBar.style.display = "none";
  status.textContent = "Done! " + indexedPhotos.length + " photos indexed. Describe the one you want.";
  searchButton.disabled = false;
}

// ---------- 3. compare two embeddings ----------
// both lists are already normalized, so multiplying and adding
// gives the cosine similarity (higher = more similar)
function similarity(a, b) {
  var total = 0;
  for (var i = 0; i < a.length; i++) {
    total = total + a[i] * b[i];
  }
  return total;
}

// ---------- 4. search ----------
async function search() {
  var description = descriptionInput.value.trim();
  if (description === "") {
    status.textContent = "Type a description first.";
    return;
  }

  status.textContent = "Searching...";

  // turn the description into an embedding too
  var textInputs = tokenizer(["a photo of " + description], { padding: true, truncation: true });
  var textOutput = await textModel(textInputs);
  var textEmbedding = textOutput.text_embeds.normalize(2, -1).data;

  // score every photo
  var scored = [];
  for (var i = 0; i < indexedPhotos.length; i++) {
    scored.push({
      url: indexedPhotos[i].url,
      score: similarity(textEmbedding, indexedPhotos[i].embedding)
    });
  }

  // best matches first
  scored.sort(function (a, b) {
    return b.score - a.score;
  });

  showResults(scored.slice(0, 9));
  status.textContent = "Top matches for: " + description;
}

// ---------- 5. show the results ----------
function showResults(matches) {
  results.innerHTML = "";

  for (var i = 0; i < matches.length; i++) {
    var card = document.createElement("div");
    card.className = "card";

    var img = document.createElement("img");
    img.src = matches[i].url;

    var label = document.createElement("p");
    label.textContent = "#" + (i + 1) + "  (score " + matches[i].score.toFixed(3) + ")";

    card.appendChild(img);
    card.appendChild(label);
    results.appendChild(card);
  }
}

// ---------- 6. choosing photos ----------
photoInput.addEventListener("change", function () {
  var files = Array.from(photoInput.files);
  if (files.length > 0) {
    indexPhotos(files);
  }
});

// Folder picker (only works in desktop Chrome and Edge)
async function getImagesFromFolder(folderHandle) {
  var files = [];
  for await (var entry of folderHandle.values()) {
    if (entry.kind === "file") {
      var file = await entry.getFile();
      if (file.type.startsWith("image/")) {
        files.push(file);
      }
    } else if (entry.kind === "directory") {
      var moreFiles = await getImagesFromFolder(entry);
      files = files.concat(moreFiles);
    }
  }
  return files;
}

if (window.showDirectoryPicker) {
  folderButton.style.display = "inline-block";

  folderButton.addEventListener("click", async function () {
    try {
      var folder = await window.showDirectoryPicker();
      status.textContent = "Reading folder...";
      var files = await getImagesFromFolder(folder);
      if (files.length > 0) {
        indexPhotos(files);
      } else {
        status.textContent = "No photos found in that folder.";
      }
    } catch (error) {
      status.textContent = "Folder access was cancelled.";
    }
  });
}

// ---------- 7. buttons ----------
searchButton.addEventListener("click", search);
descriptionInput.addEventListener("keydown", function (event) {
  if (event.key === "Enter" && !searchButton.disabled) {
    search();
  }
});

loadModels();