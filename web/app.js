const API_URL = "";

const QUESTIONS = [
    "I love exploring new ideas, books, or art.",
    "I enjoy learning foreign languages or understanding how the mind works.",
    "I plan tasks and hate leaving things unfinished.",
    "People can count on me and I follow through on what I say.",
    "I recharge by being around people because socializing energizes me.",
    "I have a lot of physical energy and love dancing or moving to music.",
    "I genuinely care about others' feelings and help when I can.",
    "I donate to causes and care about children and communities.",
    "My mood changes often and I feel things intensely.",
    "I often worry about how life is going and dwell on struggles."
];

const TRAIT_QUESTION_MAP = {
    Openness: [0, 1],
    Conscientiousness: [2, 3],
    Extraversion: [4, 5],
    Agreeableness: [6, 7],
    Neuroticism: [8, 9]
};

const GENRES = [
    "Dance", "Folk", "Country", "Classical music", "Musical", "Pop", "Rock", "Metal or Hardrock",
    "Punk", "Hiphop, Rap", "Reggae, Ska", "Swing, Jazz", "Rock n roll", "Alternative", "Latino",
    "Techno, Trance", "Opera"
];

const state = {
    index: 0,
    answers: new Array(10).fill(3)
};

const startBtn = document.getElementById("startBtn");
const quizSection = document.getElementById("quizSection");
const loadingSection = document.getElementById("loadingSection");
const resultSection = document.getElementById("resultSection");
const questionCard = document.getElementById("questionCard");
const progressText = document.getElementById("progressText");
const progressFill = document.getElementById("progressFill");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
const retryBtn = document.getElementById("retryBtn");
const traitsList = document.getElementById("traitsList");
const genresList = document.getElementById("genresList");
const explainText = document.getElementById("explainText");
const spotifyConnectBtn = document.getElementById("spotifyConnectBtn");
const spotifyPlaylistBtn = document.getElementById("spotifyPlaylistBtn");
const spotifyStatus = document.getElementById("spotifyStatus");

const SMILIES = [
    { value: 1, icon: "😞", label: "Strongly disagree" },
    { value: 2, icon: "🙁", label: "Disagree" },
    { value: 3, icon: "😐", label: "Neutral" },
    { value: 4, icon: "🙂", label: "Agree" },
    { value: 5, icon: "😄", label: "Strongly agree" }
];

let lastSubmittedAnswers = null;

startBtn.addEventListener("click", () => {
    startBtn.parentElement.classList.add("hidden");
    quizSection.classList.remove("hidden");
    renderQuestion();
});

spotifyConnectBtn.addEventListener("click", () => {
    const current = window.location.href.split("?")[0];
    const url = `${API_URL}/spotify/login-redirect?frontend_return=${encodeURIComponent(current)}`;
    window.location.href = url;
});

spotifyPlaylistBtn.addEventListener("click", async () => {
    if (!lastSubmittedAnswers) {
        spotifyStatus.textContent = "Complete the quiz first to create a playlist.";
        return;
    }

    spotifyStatus.textContent = "Creating playlist in your Spotify account...";
    try {
        const response = await fetch(`${API_URL}/spotify/create-playlist`, {
            method: "POST",
            credentials: "include",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                answers: lastSubmittedAnswers,
                top_n: 5,
                playlist_name: "PulsePersona Mix",
                is_public: false,
                limit: 30
            })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Failed to create playlist");
        }

        spotifyStatus.innerHTML = `Playlist created: <a href="${data.playlist_url}" target="_blank" rel="noopener">Open in Spotify</a>`;
    } catch (error) {
        spotifyStatus.textContent = `Spotify playlist creation failed: ${error.message}`;
    }
});

prevBtn.addEventListener("click", () => {
    if (state.index > 0) {
        state.index -= 1;
        renderQuestion();
    }
});

nextBtn.addEventListener("click", async () => {
    if (state.index < QUESTIONS.length - 1) {
        state.index += 1;
        renderQuestion();
        return;
    }

    quizSection.classList.add("hidden");
    loadingSection.classList.remove("hidden");

    const result = await fetchPrediction(state.answers);
    lastSubmittedAnswers = [...state.answers];

    loadingSection.classList.add("hidden");
    resultSection.classList.remove("hidden");
    renderResults(result);
});

retryBtn.addEventListener("click", () => {
    state.index = 0;
    state.answers = new Array(10).fill(3);
    resultSection.classList.add("hidden");
    quizSection.classList.remove("hidden");
    renderQuestion();
});

function renderQuestion() {
    const qNo = state.index + 1;
    const total = QUESTIONS.length;
    const value = state.answers[state.index];

    progressText.textContent = `Question ${qNo} / ${total}`;
    progressFill.style.width = `${(qNo / total) * 100}%`;

    const optionsHtml = SMILIES.map((item) => {
        const active = item.value === value ? "active" : "";
        return `
      <button class="smiley-btn ${active}" data-score="${item.value}" type="button" aria-label="${item.label}">
        <span class="smiley-icon">${item.icon}</span>
        <span class="smiley-label">${item.value}</span>
      </button>
    `;
    }).join("");

    questionCard.innerHTML = `
    <h3 class="question-title">Q${qNo}. ${QUESTIONS[state.index]}</h3>
    <div class="smiley-grid">
      ${optionsHtml}
    </div>
    <div id="scoreChip" class="score-chip">Selected score: ${value}</div>
    <div class="scale">
      <span>Strongly disagree</span>
      <span>Strongly agree</span>
    </div>
  `;

    prevBtn.disabled = state.index === 0;
    nextBtn.textContent = state.index === QUESTIONS.length - 1 ? "See Results" : "Next";

    const buttons = questionCard.querySelectorAll(".smiley-btn");
    const scoreChip = document.getElementById("scoreChip");

    buttons.forEach((btn) => {
        btn.addEventListener("click", () => {
            const nextValue = Number(btn.dataset.score);
            if (!nextValue) return;

            buttons.forEach((x) => x.classList.remove("active"));
            btn.classList.add("active");
            state.answers[state.index] = nextValue;
            scoreChip.textContent = `Selected score: ${nextValue}`;
        });
    });
}

async function fetchPrediction(answers) {
    try {
        const response = await fetch(`${API_URL}/predict/full`, {
            method: "POST",
            credentials: "include",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ answers, top_n: 5 })
        });

        if (!response.ok) {
            throw new Error(`API status ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        return localHeuristicPrediction(answers);
    }
}

function applySpotifyCallbackStatus() {
    const params = new URLSearchParams(window.location.search);
    const status = params.get("spotify");

    if (status === "ok") {
        spotifyStatus.textContent = "Spotify connected. You can now create your playlist.";
    } else if (status === "error") {
        const message = params.get("message") || "OAuth flow failed";
        spotifyStatus.textContent = `Spotify connection failed: ${message}`;
    }

    if (status) {
        const cleanUrl = window.location.href.split("?")[0];
        window.history.replaceState({}, "", cleanUrl);
    }
}

applySpotifyCallbackStatus();

function localHeuristicPrediction(answers) {
    const traitScores = {};
    for (const [trait, idxs] of Object.entries(TRAIT_QUESTION_MAP)) {
        traitScores[trait] = Number(((answers[idxs[0]] + answers[idxs[1]]) / 2).toFixed(2));
    }

    const openness = traitScores.Openness;
    const ext = traitScores.Extraversion;
    const agree = traitScores.Agreeableness;
    const cons = traitScores.Conscientiousness;
    const neuro = traitScores.Neuroticism;

    const base = GENRES.map((genre, i) => {
        let score = 0.25 + (i % 5) * 0.02;

        if (["Alternative", "Swing, Jazz", "Classical music", "Opera"].includes(genre)) score += openness * 0.08;
        if (["Pop", "Dance", "Techno, Trance", "Hiphop, Rap"].includes(genre)) score += ext * 0.08;
        if (["Folk", "Country", "Reggae, Ska"].includes(genre)) score += agree * 0.06;
        if (["Classical music", "Musical", "Rock n roll"].includes(genre)) score += cons * 0.05;
        if (["Metal or Hardrock", "Punk"].includes(genre)) score += neuro * 0.05;

        return {
            genre,
            probability: Math.min(0.98, Number((score / 5).toFixed(3))),
            predicted: score / 5 >= 0.5 ? 1 : 0
        };
    }).sort((a, b) => b.probability - a.probability);

    return {
        trait_scores: traitScores,
        genres: base
    };
}

function renderResults(result) {
    const traitScores = result.trait_scores || {};
    const genres = (result.genres || []).slice(0, 5);

    traitsList.innerHTML = Object.entries(traitScores)
        .map(([trait, score], idx) => {
            const width = Math.max(0, Math.min(100, (Number(score) / 5) * 100));
            return `
        <div class="trait-item" style="animation: rise-in 450ms ${idx * 90}ms ease both;">
          <span>${trait}</span>
          <div class="trait-bar"><span style="width:${width}%;"></span></div>
          <strong>${Number(score).toFixed(2)}</strong>
        </div>
      `;
        })
        .join("");

    genresList.innerHTML = genres
        .map((item, idx) => `
      <li class="genre-item" style="animation: rise-in 450ms ${idx * 90}ms ease both;">
        <strong>${item.genre}</strong>
        <div class="genre-meta">Match score: ${(Number(item.probability) * 100).toFixed(1)}%</div>
      </li>
    `)
        .join("");

    const highTrait = Object.entries(traitScores).sort((a, b) => b[1] - a[1])[0];
    const topGenre = genres[0] ? genres[0].genre : "your favorite genres";
    explainText.textContent = highTrait
        ? `Your strongest trait appears to be ${highTrait[0]} (${Number(highTrait[1]).toFixed(2)}). That profile aligns with ${topGenre} and related styles in your ranking.`
        : "Your profile is ready. Connect the backend to receive model-based explanations.";
}
