import { useState, useCallback, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { VideoPreview } from "@/components/VideoPreview";
import { DetectedObjects } from "@/components/DetectedObjects";
import { GestureDisplay } from "@/components/GestureDisplay";
import { SystemLog } from "@/components/SystemLog";
import { ControlBar } from "@/components/ControlBar";
import { useToast } from "@/hooks/use-toast";

interface DetectedObject {
  id: string;
  name: string;
  confidence: number;
  timestamp: Date;
}

interface Gesture {
  name: string;
  confidence: number;
  timestamp: Date;
}

interface LogEntry {
  // id: string; // Removed duplicate id in favor of using index in map
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  timestamp: Date;
}

const Index = () => {
  const { toast } = useToast();
  const [isActive, setIsActive] = useState(false);
  const [detectionMode, setDetectionMode] = useState<"object_detection" | "hand_gesture_recognition">("object_detection");
  const [videoFeedUrl, setVideoFeedUrl] = useState<string>("");
  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([]);
  const [lastGesture, setLastGesture] = useState<Gesture | null>(null);
  const [systemLogs, setSystemLogs] = useState<LogEntry[]>([]);
  const fetchIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const addLog = useCallback((message: string, type: LogEntry['type'] = 'info') => {
    const newLog: LogEntry = {
      // id: Date.now().toString(), // Removed id from newLog
      message,
      type,
      timestamp: new Date(),
    };
    setSystemLogs(prev => [...prev, newLog].slice(-50)); // Keep last 50 logs
  }, []);

  const handleStartCamera = useCallback(async (mode: "object_detection" | "hand_gesture_recognition") => {
    setDetectionMode(mode);
    
    try {
      // Call backend to start video processing
      const startResponse = await fetch('http://localhost:5000/start_video_processing', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ mode: mode }),
      });

      if (!startResponse.ok) {
        throw new Error(`Failed to start video processing: ${startResponse.status}`);
      }
      const startData = await startResponse.json();
      addLog(`Backend video processing ${startData.status} in ${startData.mode} mode.`, 'info');

      setIsActive(true);
      addLog(`Camera started in ${mode.replace('_', ' ')} mode`, 'success');
      
      toast({
        title: "Camera Started",
        description: `${mode.replace('_', ' ')} active`,
      });

      // Start polling for frames and data
      if (fetchIntervalRef.current) {
        clearInterval(fetchIntervalRef.current);
      }
      fetchIntervalRef.current = setInterval(async () => {
        try {
          const response = await fetch(`http://localhost:5000/video_feed`); // No mode needed in video_feed, mode is handled by streamer
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const data = await response.json();

          console.log("Fetched frame data:", data);
          if (data.frame) { // Only update if a frame is available
            setVideoFeedUrl(`data:image/jpeg;base64,${data.frame}`);
            console.log("Updated videoFeedUrl");
          } else {
            setVideoFeedUrl(""); // Clear video if no frame
          }

          if (mode === 'object_detection') {
            const newDetections: DetectedObject[] = data.detections.map((det: any) => ({
              id: Date.now().toString() + Math.random().toString(36).substring(2, 9), // Unique ID
              name: det.class,
              confidence: det.confidence,
              timestamp: new Date(),
            }));
            setDetectedObjects(newDetections);
            if (newDetections.length > 0) {
              addLog(`Detected: ${newDetections.map(d => `${d.name} (${(d.confidence * 100).toFixed(0)}%)`).join(', ')}`, 'info');
            }
          } else if (mode === 'hand_gesture_recognition') {
            if (data.gesture) {
              setLastGesture({
                name: data.gesture,
                confidence: 1, // Assuming confidence is 1 for simplicity if only name is provided
                timestamp: new Date(),
              });
              addLog(`Gesture: ${data.gesture}`, 'info');
            }
          }

        } catch (error) {
          console.error("Error fetching video feed:", error);
          addLog("Video feed error. Please check the backend server.", 'error');
          // handleStopCamera(); // Don't call handleStopCamera here directly, let the stop request handle it
          toast({
            title: "Video Feed Error",
            description: "Connection to backend lost or refused. Attempting to stop camera.",
            variant: "destructive",
          });
          // Attempt to stop the backend
          fetch('http://localhost:5000/stop_video_processing', { method: 'POST' })
            .then(res => res.json())
            .then(data => addLog(`Backend stop status: ${data.status}`, 'info'))
            .catch(err => console.error("Error stopping backend:", err));
          handleStopCamera(); // Now call handleStopCamera to clear frontend state
        }
      }, 100); // Poll every 100ms
    } catch (error) {
      console.error("Error starting video processing:", error);
      addLog("Failed to start backend video processing. Check server logs.", 'error');
      toast({
        title: "Backend Start Error",
        description: "Could not initialize video processing.",
        variant: "destructive",
      });
      setIsActive(false); // Ensure UI reflects the stopped state
    }

  }, [addLog, toast]);

  const handleStopCamera = useCallback(async () => {
    if (fetchIntervalRef.current) {
      clearInterval(fetchIntervalRef.current);
      fetchIntervalRef.current = null;
    }
    setIsActive(false);
    setVideoFeedUrl(""); // Stop video feed
    setDetectedObjects([]);
    setLastGesture(null);
    addLog("Camera stopped", 'warning');
    
    toast({
      title: "Camera Stopped", 
      description: "Detection services have been terminated",
      variant: "destructive",
    });

    try {
      const stopResponse = await fetch('http://localhost:5000/stop_video_processing', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      if (!stopResponse.ok) {
        throw new Error(`Failed to stop video processing: ${stopResponse.status}`);
      }
      const stopData = await stopResponse.json();
      addLog(`Backend video processing ${stopData.status}.`, 'info');
    } catch (error) {
      console.error("Error sending stop request to backend:", error);
      addLog("Failed to send stop signal to backend.", 'error');
    }

  }, [addLog, toast]);

  const handleExit = useCallback(() => {
    if (isActive) {
      handleStopCamera(); // Ensure camera is stopped before exit
    }
    addLog("Application exit requested", 'info');
    
    toast({
      title: "Exit Requested",
      description: "Application will close safely",
    });
  }, [isActive, handleStopCamera, addLog, toast]);

  useEffect(() => {
    // Cleanup on component unmount
    return () => {
      if (fetchIntervalRef.current) {
        clearInterval(fetchIntervalRef.current);
      }
      // Ensure the backend video processing is stopped when the component unmounts
      fetch('http://localhost:5000/stop_video_processing', { method: 'POST' })
        .then(res => res.json())
        .then(data => console.log('Backend stop on unmount:', data.status))
        .catch(err => console.error('Error stopping backend on unmount:', err));
    };
  }, []);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="border-b bg-gradient-card shadow-card"
      >
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <h1 className="text-3xl font-bold bg-gradient-primary bg-clip-text text-transparent">
                Object Detection + Gesture Recognition
              </h1>
              <p className="text-muted-foreground mt-1">
                Real-time computer vision powered by YOLO and MediaPipe
              </p>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className={`w-3 h-3 rounded-full ${
                isActive 
                  ? "bg-success animate-pulse shadow-[0_0_10px_hsl(var(--success))]" 
                  : "bg-muted"
              }`}
            />
          </div>
        </div>
      </motion.header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 min-h-[calc(100vh-200px)]">
          {/* Left Column - Video Preview */}
          <div className="lg:col-span-2">
            <VideoPreview 
              isActive={isActive}
              onStartCamera={() => handleStartCamera(detectionMode)}
              videoSource={videoFeedUrl}
            />
          </div>
          
          {/* Right Column - Detection Panels */}
          <div className="space-y-6">
            <DetectedObjects objects={detectedObjects} />
            <GestureDisplay lastGesture={lastGesture} />
            <SystemLog logs={systemLogs} />
          </div>
        </div>

        {/* Control Bar */}
        <div className="mt-6">
          <ControlBar
            isActive={isActive}
            onStartObjectDetection={() => handleStartCamera("object_detection")}
            onStartGestureRecognition={() => handleStartCamera("hand_gesture_recognition")}
            onStopCamera={handleStopCamera}
            onExit={handleExit}
          />
        </div>
      </main>
    </div>
  );
};

export default Index;
