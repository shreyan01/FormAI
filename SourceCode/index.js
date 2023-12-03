import { initializeApp } from "firebase/app";
const firebaseConfig = {
  apiKey: "AIzaSyA6KbE5VEWKaiA84dzY8jf_kEww2AdjNb0",
  authDomain: "perfex-ai.firebaseapp.com",
  projectId: "perfex-ai",
  storageBucket: "perfex-ai.appspot.com",
  messagingSenderId: "324205571455",
  appId: "1:324205571455:web:029bb0c5b1befbe1f8be4a"
};
const app = initializeApp(firebaseConfig);
// app.js

function openOverlay() {
    document.getElementById('overlay').style.display = 'block';
}
  
function closeOverlay() {
    document.getElementById('overlay').style.display = 'none';
}

const formContainer = document.getElementById('formContainer');
const loginForm = document.getElementById('loginForm');
const signupForm = document.getElementById('signupForm');

function toggleForm() {
  formContainer.style.display = formContainer.style.display === 'flex' ? 'none' : 'flex';
}

function login() {
  const email = document.getElementById('loginEmail').value;
  const password = document.getElementById('loginPassword').value;

  firebase.auth().signInWithEmailAndPassword(email, password)
    .then(userCredential => {
      console.log('Login successful:', userCredential.user);
      // Handle successful login
    })
    .catch(error => {
      console.error('Login error:', error.message);
      // Handle login error
    });
}

function signup() {
  const email = document.getElementById('signupEmail').value;
  const password = document.getElementById('signupPassword').value;

  firebase.auth().createUserWithEmailAndPassword(email, password)
    .then(userCredential => {
      console.log('Signup successful:', userCredential.user);
      // Handle successful signup
    })
    .catch(error => {
      console.error('Signup error:', error.message);
      // Handle signup error
    });
}
var modal = document.getElementById('id01');
window.onclick = function(event) {
  if (event.target == modal) {
    modal.style.display = "none";
  }
}