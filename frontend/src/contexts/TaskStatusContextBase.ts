import { createContext } from "react";

export interface TaskInfo {
  id: string;
  type: string;
  label: string;
  status: "pending" | "done" | "error";
  result?: unknown;
  error?: string;
}

export interface TaskStatusContextValue {
  tasks: TaskInfo[];
  /** timeoutMs > 0 时,超过该时长仍未完成则标记为 error 并停止轮询。 */
  startTask: (id: string, type: string, label: string, timeoutMs?: number) => void;
  dismissTask: (id: string) => void;
  creatingSessionMode: string | null;
  setCreatingSessionMode: (mode: string | null) => void;
}

const TaskStatusContext = createContext<TaskStatusContextValue | null>(null);

export default TaskStatusContext;
