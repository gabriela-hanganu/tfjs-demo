import React, { useState } from "react";
import * as tf from "@tensorflow/tfjs";
import "./ImageRecognition.css";

const MODEL_URL = "/model/model.json";
const CATEGORY_LABELS = ["truck", "bus", "motorhome", "cars"];

export default function ImageRecognition() {
  const [model, setModel] = useState(null);
  const [imageURL, setImageURL] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [allConfidences, setAllConfidences] = useState([]);

  React.useEffect(() => {
    tf.loadLayersModel(MODEL_URL).then(setModel);
  }, []);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) setImageURL(URL.createObjectURL(file));
  };

  const handlePredict = async () => {
    if (!model || !imageURL) return;
    const img = document.getElementById("input-image");
    let tensor = tf.browser.fromPixels(img)
      .resizeNearestNeighbor([224, 224])
      .toFloat()
      .expandDims();
    tensor = tensor.div(255.0);
    const preds = await model.predict(tensor).data();
    const maxIdx = preds.indexOf(Math.max(...preds));
    let category = CATEGORY_LABELS[maxIdx];
    const percent = Math.round(preds[maxIdx] * 100);
    if (percent < 10) {
    category = "Not recognized";
  }
    setPrediction(category);
    setConfidence(percent);
    setAllConfidences(Array.from(preds).map((v, i) => ({
    label: CATEGORY_LABELS[i],
    percent: Math.round(v * 100)
  })));
  };

  return (
    <div className="image-recognition-container">
      <h2 className="image-recognition-title">Image Recognition</h2>
      <input
        type="file"
        accept="image/*"
        onChange={handleImageUpload}
        className="image-upload-input"
      />
      {imageURL && (
        <div className="image-preview">
          <img
            id="input-image"
            src={imageURL}
            alt="input"
            width={224}
            height={224}
            className="image-preview-img"
          />
          <button
            onClick={handlePredict}
            className="predict-btn"
          >
            Predict
          </button>
        </div>
      )}
      {prediction && (
        <div className="result-container">
    <div className="result-category">Category: {prediction}</div>
    <div className="result-bar-bg">
      <div
        className="result-bar-fill"
        style={{ width: `${confidence}%` }}
      />
    </div>
    <div className="result-confidence">
      {confidence}% confidence
    </div>
    <div className="all-categories">
      <h4>All category predictions:</h4>
      <ul style={{ listStyle: "none", padding: 0 }}>
        {allConfidences.map(({ label, percent }) => (
          <li key={label} style={{ marginBottom: "4px" }}>
            <span style={{ fontWeight: "bold" }}>{label}:</span> {percent}%
          </li>
        ))}
      </ul>
    </div>
  </div>
      )}
    </div>
  );
}