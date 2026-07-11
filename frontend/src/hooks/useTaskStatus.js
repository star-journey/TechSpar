import { useContext } from "react";
import TaskStatusContext from "../contexts/TaskStatusContextBase";

export default function useTaskStatus() {
  return useContext(TaskStatusContext);
}
