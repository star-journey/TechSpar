import { useState, useRef, useCallback } from "react";
import { getTaskStatus } from "../api/interview";
import TaskStatusContext from "./TaskStatusContextBase";

const POLL_INTERVAL = 3000;

export function TaskStatusProvider({ children }) {
  const [tasks, setTasks] = useState([]);
  const timersRef = useRef({});

  const stopPolling = useCallback((taskId) => {
    if (timersRef.current[taskId]) {
      clearInterval(timersRef.current[taskId]);
      delete timersRef.current[taskId];
    }
  }, []);

  const startTask = useCallback((id, type, label, timeoutMs = 0) => {
    stopPolling(id);
    setTasks((prev) => {
      const filtered = prev.filter((t) => t.id !== id);
      return [...filtered, { id, type, label, status: "pending" }];
    });

    const startedAt = Date.now();
    timersRef.current[id] = setInterval(async () => {
      if (timeoutMs > 0 && Date.now() - startedAt > timeoutMs) {
        setTasks((prev) =>
          prev.map((t) =>
            t.id === id ? { ...t, status: "error", error: "处理超时，请稍后重试" } : t
          )
        );
        stopPolling(id);
        return;
      }
      try {
        const data = await getTaskStatus(id);
        if (data.status === "done" || data.status === "error") {
          setTasks((prev) =>
            prev.map((t) =>
              t.id === id ? { ...t, status: data.status, result: data.result, error: data.error } : t
            )
          );
          stopPolling(id);
        }
      } catch {
        // task not ready or network error, keep polling
      }
    }, POLL_INTERVAL);
  }, [stopPolling]);

  const dismissTask = useCallback((id) => {
    stopPolling(id);
    setTasks((prev) => prev.filter((t) => t.id !== id));
  }, [stopPolling]);

  // Global state to track interview creation across route navigations
  const [creatingSessionMode, setCreatingSessionMode] = useState(null);

  return (
    <TaskStatusContext.Provider value={{ tasks, startTask, dismissTask, creatingSessionMode, setCreatingSessionMode }}>
      {children}
    </TaskStatusContext.Provider>
  );
}
