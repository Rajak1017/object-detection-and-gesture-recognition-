import { motion } from "framer-motion";
import { Camera, Play } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

interface VideoPreviewProps {
  isActive: boolean;
  onStartCamera: () => void;
  videoSource: string;
}

export function VideoPreview({ isActive, onStartCamera, videoSource }: VideoPreviewProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="w-full h-full bg-gradient-card shadow-video rounded-2xl overflow-hidden">
        <CardContent className="p-0 h-full">
          <div className="relative w-full h-full min-h-[400px] lg:min-h-[500px] bg-gradient-video rounded-2xl flex items-center justify-center">
            {!isActive ? (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-center space-y-6"
              >
                <Camera className="w-16 h-16 text-muted-foreground mx-auto" />
                <div className="space-y-2">
                  <h3 className="text-xl font-semibold text-foreground">Camera Preview</h3>
                  <p className="text-muted-foreground">Click "Start Camera" to begin detection</p>
                </div>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={onStartCamera}
                  className="inline-flex items-center gap-2 px-6 py-3 bg-primary text-primary-foreground rounded-2xl font-medium hover:bg-primary/90 transition-smooth shadow-card"
                >
                  <Play className="w-4 h-4" />
                  Start Camera
                </motion.button>
              </motion.div>
            ) : (
              <img
                src={videoSource}
                alt="Video Feed"
                className="w-full h-full object-contain rounded-2xl"
              />
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}