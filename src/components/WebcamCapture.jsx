import React, { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam";
import { io } from "socket.io-client";

const WebcamCapture = () => {
  const webcamRef = useRef(null);
  const socketRef = useRef(null);
  const [result, setResult] = useState("Waiting for result...");
  const [borderColor, setBorderColor] = useState("gray");
  const [isCameraAllowed, setIsCameraAllowed] = useState(false);
  const [intervalId, setIntervalId] = useState(null);
  const [cameraLabel, setCameraLabel] = useState("");

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user",
  };

  const blockedCameras = [
    "Snap Camera", "OBS Virtual Camera", "ManyCam", "CamTwist", "XSplit VCam",
    "YouCam", "Ecamm Live", "vMix", "VirtualCam", "Webcamoid", "Fake Webcam",
    "AlterCam", "VidblasterX", "ChromaCam", "Logitech Capture"
  ];

  const updateBorderColor = (message) => {
    // Guard against undefined/null messages
    if (!message || typeof message !== "string") {
      setBorderColor("gray");
      return;
    }

    if (message.includes("FAKE")) {
      setBorderColor("red");
    } else if (message.includes("REAL but NOT MATCHED")) {
      setBorderColor("blue");
    } else if (message.includes("REAL")) {
      setBorderColor("#01b93b");
    } else {
      setBorderColor("gray");
    }
  };

  const sendImageToBackend = (imageSrc) => {
    if (!socketRef.current || !socketRef.current.connected) {
      console.warn('Socket not connected');
      setResult('No connection to server');
      return;
    }

    socketRef.current.emit('image', { image: imageSrc });
  };

  const captureFrame = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) {
      sendImageToBackend(imageSrc);
    }
  };

  useEffect(() => {
    // create socket connection on mount
    socketRef.current = io('http://ec2-16-171-193-102.eu-north-1.compute.amazonaws.com:5000');

    socketRef.current.on('connect', () => {
      console.info('Connected to server via websocket');
    });

    socketRef.current.on('prediction', (data) => {
      if (!data) return;
      if (data.error) {
        setResult(`Error: ${data.error}`);
        setBorderColor('gray');
        return;
      }
      // Show only the label (REAL / FAKE / FOCUS) — do not display score
      const label = data.label || 'FOCUS';
      const message = label;
      setResult(message);
      updateBorderColor(message);
    });

    const id = setInterval(() => {
      if (isCameraAllowed) {
        captureFrame();
      }
    }, 600);
    setIntervalId(id);

    return () => {
      clearInterval(id);
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, [isCameraAllowed]);

  const checkCameraAndAllow = async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === "videoinput");

      if (videoDevices.length > 0) {
        const cameraName = videoDevices[0].label; // Use the first available camera

        setCameraLabel(cameraName);

        // Check if the camera label matches any blocked virtual cameras
        const isVirtualCamera = blockedCameras.some((blockedName) =>
          cameraName.toLowerCase().includes(blockedName.toLowerCase())
        );

        if (!isVirtualCamera) {
          setIsCameraAllowed(true);
        } else {
          setIsCameraAllowed(false);
          setResult("Virtual cameras are not allowed.");
          setBorderColor("red");
        }
      } else {
        setResult("No camera found.");
        setBorderColor("gray");
      }
    } catch (error) {
      console.error("Error checking camera:", error);
      setResult("Error detecting camera.");
      setBorderColor("gray");
    }
  };

  useEffect(() => {
    checkCameraAndAllow();
  }, []);

  return (
    <div>
      <h2>Face Detection</h2>
      {isCameraAllowed ? (
        <div
          style={{
            display: "inline-block",
            border: `10px solid ${borderColor}`,
            borderRadius: "20px",
          }}
        >
          <Webcam
            audio={false}
            height={480}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            width={640}
            videoConstraints={videoConstraints}
          />
        </div>
      ) : (
        <p style={{ color: "red" }}>Virtual camera detected or no camera found. Please select a valid camera.</p>
      )}
      <p style={{ fontSize: "1.2rem", fontWeight: "600" }}>{result}</p>
      {cameraLabel && <p>Detected Camera: {cameraLabel}</p>}
    </div>
  );
};

export default WebcamCapture;








/* import React, { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam";
import axios from "axios";

const WebcamCapture = () => {
  const webcamRef = useRef(null);
  const [result, setResult] = useState("Waiting for result...");
  const [borderColor, setBorderColor] = useState("gray");
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [isVirtual, setIsVirtual] = useState(false);
  const [characteristics, setCharacteristics] = useState({});

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user",
    deviceId: selectedCamera ? selectedCamera.deviceId : undefined,
  };

  const virtualCameraList = [
    "OBS", "Snap Camera", "ManyCam", "YouCam", "XSplit", "CamTwist", "Fake Webcam", "Screenflick", "EpocCam"
  ];

  const updateBorderColor = (message) => {
    if (message.includes("FAKE")) {
      setBorderColor("red");
    } else if (message.includes("REAL but NOT MATCHED")) {
      setBorderColor("blue");
    } else if (message.includes("REAL") && message.includes("MATCHED")) {
      setBorderColor("#01b93b");
    } else {
      setBorderColor("gray");
    }
  };

  const sendImageToBackend = (imageSrc) => {
    axios
      .post("http://localhost:5000/api/process-image", {
        image: imageSrc,
      })
      .then((response) => {
        const message = response.data.message;
        setResult(message);
        updateBorderColor(message);
      })
      .catch((error) => {
        console.error("Error sending image to backend:", error);
        setResult("Error processing image.");
        setBorderColor("gray");
      });
  };

  const captureFrame = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) {
      sendImageToBackend(imageSrc);
    }
  };

  // Check for suspicious characteristics in the camera feed
  const checkCameraCharacteristics = (stream) => {
    const videoTrack = stream.getVideoTracks()[0];
    const settings = videoTrack.getSettings();

    const frameRate = settings.frameRate || 0;
    const resolution = { width: settings.width, height: settings.height };

    let isSuspicious = false;

    // Check frame rate - virtual cameras may have unusual frame rates
    if (frameRate < 15 || frameRate > 60) {
      isSuspicious = true;
    }

    // Check resolution - non-standard resolutions may indicate virtual cameras
    if (resolution.width < 640 || resolution.height < 480 || resolution.width > 1920 || resolution.height > 1080) {
      isSuspicious = true;
    }

    // Store characteristics for displaying later
    setCharacteristics({ frameRate, resolution });

    return isSuspicious;
  };

  // Fetch and analyze camera devices
  useEffect(() => {
    navigator.mediaDevices.enumerateDevices().then((devices) => {
      const videoDevices = devices.filter((device) => device.kind === "videoinput");

      // Check if selected camera is virtual or suspicious
      const suspiciousCamera = videoDevices.find((device) => {
        const lowerLabel = device.label.toLowerCase();
        return virtualCameraList.some((vc) => lowerLabel.includes(vc.toLowerCase()));
      });

      if (suspiciousCamera) {
        setIsVirtual(true);
        alert(`Virtual camera detected: ${suspiciousCamera.label}. Please switch to a real camera.`);
      } else {
        // Select the first available camera (automatically) and check characteristics
        const camera = videoDevices[0];
        setSelectedCamera(camera);

        // Get video stream to check characteristics
        navigator.mediaDevices.getUserMedia({
          video: { deviceId: camera.deviceId }
        }).then((stream) => {
          const suspicious = checkCameraCharacteristics(stream);
          if (suspicious) {
            setIsVirtual(true);
            alert("Suspicious camera characteristics detected. Please use a real camera.");
          }
        });
      }
    });
  }, []);

  useEffect(() => {
    if (!isVirtual) {
      const intervalId = setInterval(() => {
        captureFrame();
      }, 1000);
      return () => clearInterval(intervalId);
    }
  }, [isVirtual]);

  return (
    <div>
      <h2>Face Detection</h2>
      {isVirtual ? (
        <p style={{ color: "red" }}>Virtual or suspicious camera detected. Please switch to a real camera.</p>
      ) : (
        <>
          <div
            style={{
              display: "inline-block",
              border: `10px solid ${borderColor}`,
              borderRadius: "20px",
            }}
          >
            <Webcam
              audio={false}
              height={480}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              width={640}
              videoConstraints={videoConstraints}
            />
          </div>
          <p style={{ fontSize: "1.2rem", fontWeight: "600" }}>{result}</p>
          <div>
            <p><strong>Camera Characteristics:</strong></p>
            <p>Frame Rate: {characteristics.frameRate || "Unknown"}</p>
            <p>Resolution: {characteristics.resolution ? `${characteristics.resolution.width}x${characteristics.resolution.height}` : "Unknown"}</p>
          </div>
        </>
      )}
    </div>
  );
};

export default WebcamCapture; */
