import axios from "axios";

const API = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000",
  timeout: 120000,
});

export const sendMessage = async (message) => {
  const res = await API.post("/chat", {
    question: message,
  });

  return res.data;
};
