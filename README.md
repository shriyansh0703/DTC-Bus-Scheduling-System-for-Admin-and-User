🚌 DTC Bus Scheduling System for Admin and User

A full-stack web application designed to simplify and automate bus scheduling, route management, and timetable access for both administrators and users. The system provides a centralized platform for managing bus operations while allowing users to easily view route and schedule information.

📖 Overview

The DTC Bus Scheduling System addresses the challenges of manual bus scheduling and route management by providing a digital platform that streamlines operations and improves accessibility to schedule information.

The application consists of two primary modules:

Admin Module – Manage buses, routes, schedules, and system data.
User Module – View routes, search schedules, and access bus information.
✨ Features
👨‍💼 Admin Features
Secure Admin Authentication
Dashboard for System Management
Add, Update, and Delete Bus Details
Route Management
Bus Schedule Management
User Management
View and Monitor System Records
Centralized Data Control
👤 User Features
User Registration and Login
View Available Bus Routes
Search Buses by Route
Check Bus Timings and Schedules
Access Route Information
Simple and Responsive User Interface
🛠️ Tech Stack
Frontend
React.js
HTML5
CSS3
JavaScript
Backend
Node.js
Express.js
Database
MongoDB
Tools & Technologies
Git
GitHub
REST APIs
Postman
📂 Project Structure
DTC-Bus-Scheduling-System-for-Admin-and-User/
│
├── client/                 # React Frontend
│   ├── public/
│   ├── src/
│   └── package.json
│
├── server/                 # Backend Server
│   ├── controllers/
│   ├── models/
│   ├── routes/
│   ├── middleware/
│   └── package.json
│
├── README.md
│
└── .gitignore
🚀 Getting Started
Prerequisites

Make sure the following are installed on your system:

Node.js
npm
MongoDB
Installation
1. Clone the Repository
git clone https://github.com/shriyansh0703/DTC-Bus-Scheduling-System-for-Admin-and-User.git
2. Navigate to the Project Directory
cd DTC-Bus-Scheduling-System-for-Admin-and-User
3. Install Backend Dependencies
cd server
npm install
4. Install Frontend Dependencies
cd ../client
npm install
⚙️ Environment Variables

Create a .env file inside the server directory and configure the following:

PORT=5000
MONGO_URI=your_mongodb_connection_string
JWT_SECRET=your_jwt_secret
▶️ Running the Application
Start Backend Server
cd server
npm start
Start Frontend Application
cd client
npm start

The application will be available at:

Frontend: http://localhost:3000
Backend:  http://localhost:5000
📸 Screenshots
Admin Dashboard
Manage Buses
Manage Routes
Manage Schedules
View Users
User Dashboard
Search Routes
View Timetables
Access Bus Information

Add screenshots here for better project presentation.

🎯 Objectives
Digitize bus scheduling operations.
Reduce manual scheduling errors.
Improve route and timetable management.
Provide easy access to transportation information.
Enhance operational efficiency.
🔮 Future Enhancements
Real-Time Bus Tracking
GPS Integration
Live Traffic Updates
Mobile Application Support
Online Ticket Booking
Notification and Alert System
AI-Based Route Optimization
📊 Use Cases
Admin
Create and manage bus schedules.
Update route information.
Monitor system data.
Manage user records.
User
Search bus routes.
View schedules and timings.
Access route details.
Plan travel efficiently.
