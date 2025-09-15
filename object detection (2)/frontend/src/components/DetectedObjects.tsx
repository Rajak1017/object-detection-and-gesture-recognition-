import { motion, AnimatePresence } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Eye } from "lucide-react";

interface DetectedObject {
  id: string;
  name: string;
  confidence: number;
  timestamp: Date;
}

interface DetectedObjectsProps {
  objects: DetectedObject[];
}

export function DetectedObjects({ objects }: DetectedObjectsProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <Card className="h-full bg-gradient-card shadow-card rounded-2xl">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Eye className="w-5 h-5 text-primary" />
            Detected Objects
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4 pt-0">
          <div className="space-y-3 max-h-[300px] overflow-y-auto scrollbar-thin scrollbar-thumb-muted scrollbar-track-transparent">
            <AnimatePresence mode="popLayout">
              {objects.length === 0 ? (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-center py-8 text-muted-foreground"
                >
                  <div className="space-y-2">
                    <div className="w-8 h-8 border-2 border-muted-foreground/30 rounded-full mx-auto flex items-center justify-center">
                      <Eye className="w-4 h-4" />
                    </div>
                    <p className="text-sm">No objects detected yet</p>
                  </div>
                </motion.div>
              ) : (
                objects.map((object) => (
                  <motion.div
                    key={object.id}
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 10 }}
                    layout
                    className="flex items-center justify-between p-3 bg-secondary/50 rounded-xl border"
                  >
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-foreground capitalize">
                          {object.name}
                        </span>
                        <Badge variant="secondary" className="text-xs">
                          {Math.round(object.confidence * 100)}%
                        </Badge>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        {object.timestamp.toLocaleTimeString()}
                      </p>
                    </div>
                  </motion.div>
                ))
              )}
            </AnimatePresence>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}