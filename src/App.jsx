import "./App.css";
import { loadGraphModel } from "@tensorflow/tfjs-converter";
import * as tf from "@tensorflow/tfjs";
import { useEffect } from "react";
import { useState } from "react";
import { useRef } from "react";

const loadModel = async () => {
  const model = await loadGraphModel(
    "http://127.0.0.1:5500/src/graphModel/model.json",
  );
  return model;
};

const infer = async (img) => {
  const img_size = 299;
  const model = await loadModel();
  const normalized_img = tf.browser
    .fromPixels(img)
    .resizeNearestNeighbor([img_size, img_size])
    .toFloat()
    .expandDims();
  const results = model.predict(normalized_img);
  const predictions = results.arraySync();

  const classIdx = results.as1D().argMax().dataSync()[0] + 1;

  console.log(classIdx, predictions);
};

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const img = useRef();

  const handleImageChange = (e) => {
    const file = e.target.files[0];

    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage(reader.result);
      };
      reader.readAsDataURL(file);
    } else {
      setSelectedImage(null);
    }
  };

  useEffect(() => {
    if (selectedImage) {
      infer(img.current);
    }
  }, [selectedImage]);

  return (
    <div>
      <h1>Subir Imagen</h1>
      <input type="file" accept="image/*" onChange={handleImageChange} />
      <img src={selectedImage} ref={img} alt="breed" />
    </div>
  );
}

export default App;
