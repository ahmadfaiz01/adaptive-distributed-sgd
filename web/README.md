# Adaptive Distributed SGD Dashboard

This is the Next.js frontend for the distributed SGD project. It loads static CSV files from `public/data/` and turns them into an interactive dashboard for the final demo.

## Local Run

```powershell
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

## Production Build

```powershell
npm run build
```

## Deploy To Vercel

From this `web/` directory:

```powershell
npx vercel login
npx vercel --prod
```

Or with a token:

```powershell
npx vercel --prod --token YOUR_VERCEL_TOKEN
```

## Data Refresh

After rerunning the Python simulator, refresh dashboard data by copying:

```powershell
Copy-Item ..\results\raw\summary.csv public\data\summary.csv -Force
Copy-Item ..\results\raw\steps.csv public\data\steps.csv -Force
Copy-Item ..\results\figures\improvement_table.csv public\data\improvement_table.csv -Force
```
