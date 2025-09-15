import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Camera, Hand, Square, X } from "lucide-react";

interface ControlBarProps {
  isActive: boolean;
  onStartObjectDetection: () => void;
  onStartGestureRecognition: () => void;
  onStopCamera: () => void;
  onExit: () => void;
}

export function ControlBar({ isActive, onStartObjectDetection, onStartGestureRecognition, onStopCamera, onExit }: ControlBarProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.6 }}
      className="w-full"
    >
      <div className="bg-gradient-card shadow-card rounded-2xl p-4 border">
        <div className="flex items-center justify-center gap-4 flex-wrap">
          {!isActive ? (
            <>
              <motion.div
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Button
                  variant="success"
                  size="lg"
                  onClick={onStartObjectDetection}
                  className="rounded-2xl shadow-sm font-semibold"
                >
                  <Camera className="w-5 h-5" />
                  Start Object Detection
                </Button>
              </motion.div>
              <motion.div
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Button
                  variant="success"
                  size="lg"
                  onClick={onStartGestureRecognition}
                  className="rounded-2xl shadow-sm font-semibold"
                >
                  <Hand className="w-5 h-5" />
                  Start Gesture Recognition
                </Button>
              </motion.div>
            </>
          ) : (
            <motion.div
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Button
                variant="destructive"
                size="lg"
                onClick={onStopCamera}
                className="rounded-2xl shadow-sm font-semibold"
              >
                <Square className="w-5 h-5" />
                Stop Camera
              </Button>
            </motion.div>
          )}
          
          <motion.div
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <Button
              variant="outline"
              size="lg"
              onClick={onExit}
              className="rounded-2xl shadow-sm font-semibold"
            >
              <X className="w-5 h-5" />
              Exit
            </Button>
          </motion.div>
        </div>
      </div>
    </motion.div>
  );
}