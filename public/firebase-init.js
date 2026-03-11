// =====================================================================
// SETUP REQUIRED before online multiplayer will work:
//
//  1. Go to Firebase Console (console.firebase.google.com)
//     → Select project "oasis-horse-game"
//     → Build → Realtime Database → Create database
//     → Choose your region → Start in test mode (update rules later)
//
//  2. Go to Project Settings → General → Your apps
//     → Click the web app (</>) icon to register an app if none exists
//     → Copy the firebaseConfig object and replace the values below
//
//  3. Deploy database rules:  firebase deploy --only database
//
// =====================================================================

const firebaseConfig = {
  apiKey:             "AIzaSyBaLXTn26-NiZ3q-hUDDayudvse8aepbu0",
  authDomain:         "oasis-horse-game.firebaseapp.com",
  databaseURL:        "https://oasis-horse-game-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId:          "oasis-horse-game",
  storageBucket:      "oasis-horse-game.firebasestorage.app",
  messagingSenderId:  "676146386771",
  appId:              "1:676146386771:web:0d630863f7af83733df3a6"
};

try {
  firebase.initializeApp(firebaseConfig);
  window.db = firebase.database();
  window.auth = firebase.auth(); // Make auth available globally
} catch (e) {
  console.warn('[Multiplayer] Firebase init failed — online play unavailable.', e);
  window.db = null;
  window.auth = null;
}
