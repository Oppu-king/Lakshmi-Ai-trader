const chatBox = document.getElementById("chat-box");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const moodDisplay = document.getElementById("mood");

// --- Chat Logic ---
chatForm.onsubmit = function (e) {
  e.preventDefault();
  const msg = userInput.value;
  chatBox.innerHTML += `<p><strong>You:</strong> ${msg}</p>`;
  userInput.value = "";

  fetch("/chat", {
    method: "POST",
    body: new URLSearchParams({ message: msg })
  })
    .then(res => res.json())
    .then(data => {
      chatBox.innerHTML += `<p><strong>Lakshmi:</strong> ${data.reply}</p>`;
      moodDisplay.textContent = data.mood;
      chatBox.scrollTop = chatBox.scrollHeight;
    });
};

function quickSay(msg) {
  userInput.value = msg;
  chatForm.dispatchEvent(new Event("submit"));
}

// --- Voice Input ---
function startVoice() {
  const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
  recognition.lang = "en-IN";
  recognition.start();
  recognition.onresult = (event) => {
    userInput.value = event.results[0][0].transcript;
    chatForm.dispatchEvent(new Event("submit"));
  };
}

// --- Manual LTP Update ---
document.getElementById("manual-ltp-form").onsubmit = function (e) {
  e.preventDefault();
  fetch("/update_manual_ltp", {
    method: "POST",
    body: new FormData(this)
  }).then(() => {
    alert("Manual LTP updated");
    updatePrice();
  });
};

// --- Target Price Update ---
document.getElementById("target-form").onsubmit = function (e) {
  e.preventDefault();
  fetch("/update_targets", {
    method: "POST",
    body: new FormData(this)
  }).then(() => alert("Targets updated"));
};

// --- Signal Setup ---
document.getElementById("signal-form").onsubmit = function (e) {
  e.preventDefault();
  fetch("/set_signal", {
    method: "POST",
    body: new FormData(this)
  }).then(() => alert("Signal saved"));
};

// --- Love Diary Submit ---
document.getElementById("diary-form").onsubmit = function (e) {
  e.preventDefault();
  fetch("/save_diary", {
    method: "POST",
    body: new FormData(this)
  }).then(() => {
    alert("❤️ Saved to Love Diary!");
    this.reset();
  });
};

// --- Voice Upload ---
document.getElementById("voice-form").onsubmit = function (e) {
  e.preventDefault();
  fetch("/upload_voice", {
    method: "POST",
    body: new FormData(this)
  }).then(() => {
    alert("🎤 Voice uploaded!");
    this.reset();
    loadVoiceList();
  });
};

// --- Load Voices ---
function loadVoiceList() {
  fetch("/voice_list")
    .then(res => res.json())
    .then(files => {
      const voiceList = document.getElementById("voice-list");
      let html = ``;
      if (files.length === 0) {
        html += `<p>No voice entries yet.</p>`;
      } else {
        files.reverse().forEach(file => {
          html += `
            <p>
              <audio controls preload="auto">
                <source src="/static/voice_notes/${file}" type="audio/mpeg">
                Not supported.
              </audio>
              <br/><small>${file}</small>
            </p>
          `;
        });
      }
      voiceList.innerHTML = html;
    });
}

// --- LTP Updater ---
function updatePrice() {
  fetch("/get_price")
    .then(res => res.json())
    .then(data => {
      document.getElementById("ltp").innerText = data.ltp;
      document.getElementById("status").innerText = data.status;
      if (data.status.includes("Hit")) {
        document.getElementById("alertSound").play();
      }
    });
}

// --- Trade Analyzer (Phase 13) ---
document.getElementById("analyzer-form").onsubmit = function (e) {
  e.preventDefault();
  fetch("/analyze_strategy", {
    method: "POST",
    body: new FormData(this)
  })
    .then(res => res.json())
    .then(data => {
      document.getElementById("analyzer-result").innerHTML = `
        🧠 Verdict: <b>${data.verdict}</b><br>
        📊 R:R Ratio: <b>${data.rr_ratio}</b><br>
        💬 ${data.comment}
      `;
    });
};

// Initial Load
setInterval(updatePrice, 5000);
updatePrice();
loadVoiceList()
