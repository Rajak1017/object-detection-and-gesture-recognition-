import { motion, AnimatePresence } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Terminal, Info, CheckCircle, AlertCircle } from "lucide-react";

interface LogEntry {
  // id: string; // Removed id from interface
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  timestamp: Date;
}

interface SystemLogProps {
  logs: LogEntry[];
}

const getLogIcon = (type: LogEntry['type']) => {
  switch (type) {
    case 'success':
      return <CheckCircle className="w-4 h-4 text-success" />;
    case 'warning':
      return <AlertCircle className="w-4 h-4 text-yellow-500" />;
    case 'error':
      return <AlertCircle className="w-4 h-4 text-destructive" />;
    default:
      return <Info className="w-4 h-4 text-primary" />;
  }
};

const getLogBadgeVariant = (type: LogEntry['type']) => {
  switch (type) {
    case 'success':
      return 'default';
    case 'warning':
      return 'secondary';
    case 'error':
      return 'destructive';
    default:
      return 'outline';
  }
};

export function SystemLog({ logs }: SystemLogProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, delay: 0.4 }}
    >
      <Card className="bg-gradient-card shadow-card rounded-2xl">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Terminal className="w-5 h-5 text-primary" />
            System Log
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4 pt-0">
          <div className="space-y-2 max-h-[250px] overflow-y-auto scrollbar-thin scrollbar-thumb-muted scrollbar-track-transparent">
            <AnimatePresence mode="popLayout">
              {logs.length === 0 ? (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-center py-6 text-muted-foreground"
                >
                  <div className="space-y-2">
                    <div className="w-8 h-8 border-2 border-muted-foreground/30 rounded-full mx-auto flex items-center justify-center">
                      <Terminal className="w-4 h-4" />
                    </div>
                    <p className="text-sm">No system messages</p>
                  </div>
                </motion.div>
              ) : (
                logs.slice().reverse().map((log, index) => (
                  <motion.div
                    key={index} // Changed key to index
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 10 }}
                    layout
                    className="flex items-start gap-3 p-2 rounded-lg hover:bg-secondary/30 transition-colors"
                  >
                    <div className="mt-0.5">
                      {getLogIcon(log.type)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <Badge 
                          variant={getLogBadgeVariant(log.type)}
                          className="text-xs px-2 py-0.5"
                        >
                          {log.type.toUpperCase()}
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          {log.timestamp.toLocaleTimeString()}
                        </span>
                      </div>
                      <p className="text-sm text-foreground leading-relaxed">
                        {log.message}
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