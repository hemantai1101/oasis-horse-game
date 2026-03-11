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
  apiKey:            'REPLACE_WITH_YOUR_API_KEY',
  authDomain:        'oasis-horse-game.firebaseapp.com',
  databaseURL:       'https://oasis-horse-game-default-rtdb.firebaseio.com',
  projectId:         'oasis-horse-game',
  storageBucket:     'oasis-horse-game.appspot.com',
  messagingSenderId: 'REPLACE_WITH_SENDER_ID',
  appId:             'REPLACE_WITH_APP_ID',
};

try {
  firebase.initializeApp(firebaseConfig);
  window.db = firebase.database();
} catch (e) {
  console.warn('[Multiplayer] Firebase init failed — online play unavailable.', e);
  window.db = null;
}
