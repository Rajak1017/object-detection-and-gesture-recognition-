import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Hand } from "lucide-react";

interface Gesture {
  name: string;
  confidence: number;
  timestamp: Date;
}

interface GestureDisplayProps {
  lastGesture: Gesture | null;
}

export function GestureDisplay({ lastGesture }: GestureDisplayProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
    >
      <Card className="bg-gradient-card shadow-card rounded-2xl">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Hand className="w-5 h-5 text-primary" />
            Last Gesture
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4 pt-0">
          {lastGesture ? (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="space-y-3"
            >
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold text-foreground capitalize">
                  {lastGesture.name}
                </span>
                <Badge 
                  variant={lastGesture.confidence > 0.8 ? "default" : "secondary"}
                  className="text-sm font-semibold"
                >
                  {Math.round(lastGesture.confidence * 100)}%
                </Badge>
              </div>
              
              <div className="w-full bg-secondary rounded-full h-2">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${lastGesture.confidence * 100}%` }}
                  transition={{ duration: 0.8, ease: "easeOut" }}
                  className="h-2 bg-gradient-primary rounded-full"
                />
              </div>
              
              <p className="text-xs text-muted-foreground">
                Detected at {lastGesture.timestamp.toLocaleTimeString()}
              </p>
            </motion.div>
          ) : (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center py-6 text-muted-foreground"
            >
              <div className="space-y-2">
                <div className="w-8 h-8 border-2 border-muted-foreground/30 rounded-full mx-auto flex items-center justify-center">
                  <Hand className="w-4 h-4" />
                </div>
                <p className="text-sm">No gestures detected</p>
              </div>
            </motion.div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}