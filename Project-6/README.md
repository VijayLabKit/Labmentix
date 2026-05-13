# Summity AI

Summity AI is a full-stack AI text summarization platform developed by **VijayLab Kit** under **Labmentix**. The application allows users to paste text or upload PDF documents, process the content through a backend API, and generate structured summaries using Gemini AI.

## Project Information

- **Project Name:** Summity AI
- **Developer:** Ishan Chowdhury
- **Organization:** Labmentix
- **Project Type:** AI-Based Text Summarization Platform
- **Frontend:** React, Vite, Tailwind CSS
- **Backend:** Node.js, Express.js
- **AI Engine:** Gemini AI
- **PDF Processing:** Multer and pdf-parse

## Features

- Paste long text and generate summaries
- Upload PDF files for summarization
- Generate summaries using Gemini AI
- Choose summary length: short, medium, or long
- Select output style: paragraph or bullet points
- Generate summaries in multiple languages
- Extract important keywords
- Copy generated summary
- Download summary as a TXT file
- Responsive UI for mobile, tablet, and desktop
- Backend-based API key protection
- Long document chunking support
- Clear error handling for invalid input and unsupported PDFs

## Project Structure

```txt
summity-ai/
├── backend/
│   ├── routes/
│   │   └── summarize.js
│   ├── utils/
│   │   └── extractPdfText.js
│   ├── .env.example
│   ├── package.json
│   └── server.js
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   └── vite.config.js
│
├── .gitignore
└── README.md
```

## How It Works

The application follows this workflow:

```txt
User Input → React Frontend → Express Backend → Gemini AI → Summary Output
```

The frontend collects text or PDF input from the user. The backend receives the request, extracts PDF text if needed, sends the content to Gemini AI, and returns a structured summary with keywords.

## Backend Setup

Go to the backend folder:

```bash
cd backend
```

Install dependencies:

```bash
npm install
```

Create a `.env` file inside the backend folder:

```env
GEMINI_API_KEY=your_gemini_api_key_here
PORT=5000
CLIENT_ORIGIN=http://localhost:5173
GEMINI_MODEL=gemini-2.5-flash
```

Start the backend server:

```bash
npm run dev
```

The backend will run on:

```txt
http://localhost:5000
```

Health check endpoint:

```txt
http://localhost:5000/health
```

## Frontend Setup

Open another terminal and go to the frontend folder:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the frontend:

```bash
npm run dev
```

The frontend will run on:

```txt
http://localhost:5173
```

## Environment Variables

Backend environment variables:

```env
GEMINI_API_KEY=your_gemini_api_key_here
PORT=5000
CLIENT_ORIGIN=http://localhost:5173
GEMINI_MODEL=gemini-2.5-flash
```

Frontend environment variable for live deployment:

```env
VITE_BACKEND_URL=https://your-backend-url.onrender.com/api/summarize
```

## Deployment

Recommended free deployment platforms:

- **Frontend:** Vercel
- **Backend:** Render

For Vercel frontend deployment:

```txt
Root Directory: frontend
Build Command: npm run build
Output Directory: dist
Install Command: npm install
```

For Render backend deployment:

```txt
Root Directory: backend
Build Command: npm install
Start Command: npm start
```

## Important Notes

- Do not upload `.env` files to GitHub.
- Store the Gemini API key only in backend environment variables.
- Scanned or image-only PDFs are not supported because OCR is not included.
- Very large documents may take longer because they are processed in chunks.
- Render free services may take time to wake up after inactivity.

## Author

Developed by **VijayLab Kit**  
Under **Labmentix**

## License

This project is created for educational and project demonstration purposes.
