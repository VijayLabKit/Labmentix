import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import summarizeRouter from "./routes/summarize.js";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;
const CLIENT_ORIGIN = process.env.CLIENT_ORIGIN || "http://localhost:5173";

app.use(
  cors({
    origin: CLIENT_ORIGIN,
    methods: ["GET", "POST"],
    allowedHeaders: ["Content-Type"]
  })
);

app.use(express.json({ limit: "20mb" }));
app.use(express.urlencoded({ extended: true, limit: "20mb" }));

app.use("/api/summarize", summarizeRouter);

app.get("/health", (req, res) => {
  res.status(200).json({
    status: "ok",
    service: "Summity AI backend"
  });
});

app.use((err, req, res, next) => {
  if (err.code === "LIMIT_FILE_SIZE") {
    return res.status(400).json({
      error: "PDF file is too large. Please upload a PDF smaller than 10MB."
    });
  }

  if (err.message === "Only PDF files are allowed.") {
    return res.status(400).json({
      error: "Only PDF files are allowed."
    });
  }

  console.error("Server error:", err);
  res.status(500).json({
    error: "Something went wrong on the server."
  });
});

app.listen(PORT, () => {
  console.log(`Summity AI backend running on http://localhost:${PORT}`);
});