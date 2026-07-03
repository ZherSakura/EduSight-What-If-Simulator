// Local dev (docker compose / http.server) talks to the local backend;
// the deployed site talks to the Render backend.
const API_BASE_URL =
  ["localhost", "127.0.0.1"].includes(window.location.hostname)
    ? "http://localhost:8000"
    : "https://edusight-backend-s2oa.onrender.com";

let selectedStudentId = "STU-0001";

let currentStudent = {
  id: "STU-0001",
  name: "Muhammad Ali bin Faisal",
  form: "Form 4 Jujur",
  school: "SMK Damansara Utama",
  currentScore: 89,
  currentRisk: "High Risk"
};

let defaultInputs = {
  attendance: 62,
  grades: 60,
  counselling: 0,
  welfare: 0
};

let riskFactorChart = null;

document.addEventListener("DOMContentLoaded", () => {
  setupSliderListeners();
  setupStudentSearch();
  setupSpeech();

  displayStudentProfile(currentStudent);

  loadStudentData(selectedStudentId);
});

function setupStudentSearch() {
  const searchInput = document.getElementById("studentSearchInput");

  if (!searchInput) return;

  searchInput.addEventListener("keydown", event => {
    if (event.key === "Enter") {
      const studentId = searchInput.value.trim();

      if (!studentId) {
        Swal.fire({
          icon: "warning",
          title: "Missing student ID",
          text: "Please enter a student ID before searching.",
          confirmButtonColor: "#3b82f6"
        });
        return;
      }

      selectedStudentId = studentId;
      loadStudentData(selectedStudentId);
    }
  });
}

async function loadStudentData(studentId = selectedStudentId) {
  try {
    const response = await fetch(`${API_BASE_URL}/students/${studentId}`);

    if (!response.ok) {
      throw new Error("Student API failed");
    }

    const data = await response.json();

  currentStudent = {
    id: data.id || data.student_id || studentId,
    name: data.name,
    form: data.form || data.grade,
    school: data.school,
    currentScore: data.currentScore ?? data.current_score,
    currentRisk: data.currentRisk ?? data.current_risk,
    attendance: data.attendance,
    grades: data.grades,
    counselling: data.counselling ?? 0,
    welfare: data.welfare ?? 0
  };

    displayStudentProfile(currentStudent);
  } catch (error) {
    console.warn("Student API unavailable. Using mock student.");

    const mockStudent = getMockStudentById(studentId);

    if (!mockStudent) {
      Swal.fire({
        icon: "error",
        title: "Student not found",
        text: `No student found with ID "${studentId}".`,
        confirmButtonColor: "#3b82f6"
      });
      return;
    }

    currentStudent = mockStudent;
    displayStudentProfile(currentStudent);
  }

  defaultInputs = {
    attendance: currentStudent.attendance ?? 62,
    grades: currentStudent.grades ?? 60,
    counselling: currentStudent.counselling ?? 0,
    welfare: currentStudent.welfare ?? 0
  };

  resetSlidersToStudentDefault();
  const defaultResult = generateDefaultResult();
  updateSimulationUI(defaultResult);
  updateRecommendations(defaultResult);
}

function resetSlidersToStudentDefault() {
  document.getElementById("attendance").value = defaultInputs.attendance;
  document.getElementById("grades").value = defaultInputs.grades;
  document.getElementById("counselling").value = defaultInputs.counselling;
  document.getElementById("welfare").value = defaultInputs.welfare;

  updateSliderLabels(defaultInputs);
}

function getMockStudentById(studentId) {
  const mockStudents = {
    "STU-001": {
      id: "STU-001",
      name: "Muhammad Ali bin Faisal",
      form: "Form 4 Jujur",
      school: "SMK Damansara Utama",
      currentScore: 89,
      currentRisk: "High Risk",
      attendance: 62,
      grades: 60,
      counselling: 0,
      welfare: 0
    },
    "STU-002": {
      id: "STU-002",
      name: "Aisyah Binti Rahman",
      form: "Form 5 Amanah",
      school: "SMK Damansara Utama",
      currentScore: 62,
      currentRisk: "Medium Risk",
      attendance: 78,
      grades: 65,
      counselling: 0,
      welfare: 0
    },
    "STU-003": {
      id: "STU-003",
      name: "Daniel Tan Wei Ming",
      form: "Form 3 Bestari",
      school: "SMK Damansara Utama",
      currentScore: 35,
      currentRisk: "Low Risk",
      attendance: 90,
      grades: 80,
      counselling: 0,
      welfare: 0
    }
  };

  return mockStudents[studentId] || null;
}

function displayStudentProfile(data) {
  document.getElementById("studentName").textContent = data.name;
  document.getElementById("studentInfo").textContent =
    `✥ ${data.form}   ⟟ ${data.school}`;
  document.getElementById("studentInitials").textContent = getInitials(data.name);
  document.getElementById("currentRiskLevel").textContent = `▲ ${data.currentRisk}`;
}

function getInitials(name) {
  return name
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .map(word => word[0])
    .join("")
    .toUpperCase();
}

function setupSliderListeners() {
  document.querySelectorAll("input[type='range']").forEach(input => {
    input.addEventListener("input", () => {
      updateSliderLabels(getSimulationInputs());
    });

    input.addEventListener("change", runSimulation);
  });
}

function getSimulationInputs() {
  return {
    attendance: Number(document.getElementById("attendance").value),
    grades: Number(document.getElementById("grades").value),
    counselling: Number(document.getElementById("counselling").value),
    welfare: Number(document.getElementById("welfare").value)
  };
}

function updateSliderLabels(inputs) {
  document.getElementById("attendanceValue").textContent = `${inputs.attendance}%`;
  document.getElementById("gradesValue").textContent = inputs.grades;
  document.getElementById("counsellingValue").textContent =
    `${inputs.counselling} Sessions`;

  const welfareLabels = ["None", "Partial", "Full"];
  document.getElementById("welfareValue").textContent = welfareLabels[inputs.welfare];
}

async function runSimulation() {
  const inputs = getSimulationInputs();
  updateSliderLabels(inputs);

  const simulationPayload = {
    studentId: currentStudent.id,
    inputs
  };

  console.log("Sending simulation input to cloud:", simulationPayload);

  try {
    const response = await fetch(`${API_BASE_URL}/simulate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(simulationPayload)
    });

    if (!response.ok) {
      throw new Error("Simulation API failed");
    }

    const cloudResult = await response.json();

    console.log("Received result from cloud:", cloudResult);

    const finalResult = normalizeSimulationResult(cloudResult);

    updateSimulationUI(finalResult);
    updateRecommendations(finalResult);

  } catch (error) {
    console.warn("Backend unavailable. Using mock simulation.");

    const mockResult = generateMockSimulationResult(inputs);

    updateSimulationUI(mockResult);
    updateRecommendations(mockResult);
  }
}

// Speech Synthesis API
let isSpeaking = false;
let availableVoices = [];

function setupSpeech() {
  if (!("speechSynthesis" in window)) return;

  loadVoices();
  if (typeof speechSynthesis.onvoiceschanged !== "undefined") {
    speechSynthesis.onvoiceschanged = loadVoices;
  }
}

function loadVoices() {
  availableVoices = window.speechSynthesis
    .getVoices()
    .filter(v => v.lang && v.lang.toLowerCase().startsWith("en"));
}

function pickFemaleVoice() {
  const femaleIdx = availableVoices.findIndex(v =>
    /zira|female|samantha|karen|serena|susan|victoria|allison|kate|tessa|moira/i.test(v.name)
  );
  if (femaleIdx !== -1) return availableVoices[femaleIdx];
  return availableVoices[0] || null;
}

function toggleSpeakNarrative() {
  if (!("speechSynthesis" in window)) {
    Swal.fire({
      icon: "info",
      title: "Speech not supported",
      text: "Your browser doesn't support the Speech Synthesis API."
    });
    return;
  }

  if (isSpeaking) {
    window.speechSynthesis.cancel();
    setSpeakingState(false);
    return;
  }

  const narrative = document.getElementById("probabilityMessage").textContent.trim();
  const projected = document.getElementById("projectedRiskText").textContent.trim();
  const score = document.getElementById("riskScore").textContent.trim();

  const phrase = `${currentStudent.name}. Projected risk score: ${score}. ${projected}. ${narrative}`;

  const utterance = new SpeechSynthesisUtterance(phrase);
  utterance.rate = 1.0;
  utterance.pitch = 1.0;

  const voice = pickFemaleVoice();
  if (voice) {
    utterance.voice = voice;
    utterance.lang = voice.lang;
  } else {
    utterance.lang = "en-US";
  }

  utterance.onend = () => setSpeakingState(false);
  utterance.onerror = () => setSpeakingState(false);

  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
  setSpeakingState(true);
}

function setSpeakingState(speaking) {
  isSpeaking = speaking;
  const icon = document.getElementById("speakIcon");
  const label = document.getElementById("speakLabel");
  if (icon) icon.textContent = speaking ? "⏹️" : "🔊";
  if (label) label.textContent = speaking ? "Stop" : "Read report aloud";
}

// Youtube Player API
const YT_VIDEO_ID = "F23ak31YnTI";

let ytPlayer;

function onYouTubeIframeAPIReady() {
  ytPlayer = new YT.Player("ytPlayer", {
    height: "315",
    width: "100%",
    videoId: YT_VIDEO_ID,
    playerVars: {
      rel: 0,
      modestbranding: 1,
      playsinline: 1
    },
    events: {
      onReady: onYtPlayerReady,
      onStateChange: onYtStateChange,
      onError: onYtError
    }
  });
}

function onYtPlayerReady() {
  setYtStatus("Ready");
}

function onYtStateChange(event) {
  const states = {
    [-1]: "Unstarted",
    [0]: "Ended",
    [1]: "Playing",
    [2]: "Paused",
    [3]: "Buffering",
    [5]: "Cued"
  };
  setYtStatus(states[event.data] || "");
}

function onYtError(event) {
  setYtStatus(`Error (code ${event.data})`);
}

function setYtStatus(text) {
  const el = document.getElementById("ytStatus");
  if (el) el.textContent = text;
}

function playVideo() {
  if (ytPlayer && ytPlayer.playVideo) ytPlayer.playVideo();
}

function pauseVideo() {
  if (ytPlayer && ytPlayer.pauseVideo) ytPlayer.pauseVideo();
}

function loadVideo() {
  if (ytPlayer && ytPlayer.loadVideoById) ytPlayer.loadVideoById(YT_VIDEO_ID);
}

function normalizeSimulationResult(result) {
  const rawWeights = result.weights ?? result.factor_weights ?? {};

  return {
    baselineScore:
      result.baselineScore ?? result.baseline_score,

    projectedScore:
      result.projectedScore ?? result.projected_score,

    riskLevel:
      result.riskLevel ?? result.risk_level,

    riskLabel:
      result.riskLabel ?? result.risk_label,

    scoreChangeText:
      result.scoreChangeText ?? result.score_change_text,

    dropoutProbability:
      result.dropoutProbability ??
      result.dropout_probability,

    insightText:
      result.insightText ??
      result.insight_text,

    narrative:
      result.narrative ?? "",

    weights: {
      attendance:
        rawWeights.attendance ?? 0,

      academic:
        rawWeights.academic ??
        rawWeights.academic_performance ??
        0,

      socioeconomic:
        rawWeights.socioeconomic ??
        rawWeights.socioeconomic_status ??
        0,

      family:
        rawWeights.family ??
        rawWeights.family_support ??
        0
    },

    recommendations:
      result.recommendations ?? []
  };
}

function updateSimulationUI(result) {
  const score = result.projectedScore;
  const riskColor = getRiskColor(score);

  document.getElementById("riskScore").textContent = score;
  document.getElementById("riskScore").style.color = riskColor;

  document.getElementById("scoreBadge").textContent = result.scoreChangeText;
  document.getElementById("projectedRiskText").textContent = result.riskLevel;
  document.getElementById("currentScore").textContent = result.baselineScore;

  document.getElementById("projectedScore").textContent =
    `${score} - ${result.riskLabel}`;

  document.getElementById("projectedLabel").textContent = `PROJECTED : ${score}`;
  document.getElementById("projectedLabel").style.color = riskColor;

  document.getElementById("spectrumMarker").style.left = `${score}%`;

  document.getElementById("dropoutProbability").textContent =
    result.dropoutProbability === "-"
      ? "-%"
      : `${result.dropoutProbability}%`;

  document.getElementById("probabilityMessage").textContent =
    result.narrative || result.probabilityText || "";

  document.getElementById("insightText").textContent =
    result.insightText;

  updateWeights(result.weights);
}

function updateWeights(weights) {
  const chartData = [
    weights.attendance ?? 0,
    weights.academic ?? 0,
    weights.socioeconomic ?? 0,
    weights.family ?? 0
  ];

  const ctx = document.getElementById("riskFactorChart");

  if (!ctx) return;

  if (!riskFactorChart) {
    riskFactorChart = new Chart(ctx, {
      type: "bar",
      plugins: [ChartDataLabels],

      data: {
        labels: ["Attendance", "Academic", "Socioeconomic", "Family"],
        datasets: [{
          label: "Risk Weight (%)",
          data: chartData,
          backgroundColor: ["#ff3d3d", "#ffd85c", "#6ca7ff", "#a7f542"],
          borderWidth: 0,
          barThickness: 20
        }]
      },

      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,

        plugins: {
          legend: { display: false },

          tooltip: {
            callbacks: {
              label: function(context) {
                return `${context.raw}%`;
              }
            }
          },

          datalabels: {
            anchor: "end",
            align: "right",
            formatter: function(value) {
              return value + "%";
            },
            color: "#374151",
            font: {
              size: 12,
              weight: "bold"
            }
          }
        },

        scales: {
          x: {
            beginAtZero: true,
            max: 100,
            ticks: {
              callback: function(value) {
                return value + "%";
              }
            }
          },
          y: {
            grid: { display: false }
          }
        }
      }
    });
  } else {
    riskFactorChart.data.datasets[0].data = chartData;
    riskFactorChart.update();
  }
}

function updateRecommendations(result) {
  const box = document.getElementById("recommendations");
  if (!box) return;

  const urgencyPriority = {
    critical: 4,
    high: 3,
    medium: 2,
    routine: 1
  };

  const recommendations = (result.recommendations || [])
    .sort((a, b) => {
      return (
        (urgencyPriority[b.urgency] || 0) -
        (urgencyPriority[a.urgency] || 0)
      );
    })
    .slice(0, 3);

  if (recommendations.length === 0) {
    box.innerHTML = `
      <div class="empty-recommendation">
        <h4>No recommendations available.</h4>
        <p>Run a simulation to generate intervention recommendations.</p>
      </div>
    `;
    return;
  }

  box.innerHTML = "";

  recommendations.forEach(rec => {
    const card = document.createElement("div");

    card.className = `recommend-card ${
      rec.urgency === "critical"
        ? "red"
        : rec.urgency === "high"
        ? "yellow"
        : rec.urgency === "medium"
        ? "blue"
        : "green"
    }`;

    card.innerHTML = `
      <span>${getRecommendationIcon(rec.category)}</span>
      <h4>${rec.title || "Recommended Action"}</h4>
      <p>${rec.description || "Action is recommended."}</p>
      <small>${rec.expected_impact || ""}</small>
    `;

    box.appendChild(card);
  });

  if (result.narrative) {
    document.getElementById("probabilityMessage").textContent =
      result.narrative;
  }
}

function updateMockRecommendations() {
  const mockCloudResult = {
    recommendations: [
      {
        category: "attendance",
        urgency: "critical",
        title: "Attendance Improvement Plan (AIP)",
        description:
          "Worsening attendance trend detected. Create a formal Attendance Improvement Plan with weekly check-ins.",
        expected_impact: "stabilise trend, reduce risk by ~5 pts",
        owner: "Discipline Teacher"
      },
      {
        category: "academic",
        urgency: "high",
        title: "Peer Tutoring / Additional Support",
        description:
          "Academic score below passing benchmark. Pair with a peer tutor or assign supplemental exercises.",
        expected_impact: "reduce risk score by ~5–8 pts",
        owner: "Subject Teachers"
      },
      {
        category: "counselling",
        urgency: "medium",
        title: "Begin Structured Counselling Programme",
        description:
          "Schedule at least 3 counselling sessions to reduce risk by ~8 points.",
        expected_impact: "reduce risk score by ~8 pts",
        owner: "School Counsellor"
      },
      // {
      //   category: "welfare",
      //   urgency: "high",
      //   title: "Welfare Assistance Referral",
      //   description:
      //     "Refer student to welfare programmes such as RMT meal assistance and book aid schemes.",
      //   expected_impact: "reduce socioeconomic burden and improve retention",
      //   owner: "School Welfare Officer"
      // },
      // {
      //   category: "family",
      //   urgency: "medium",
      //   title: "Family Engagement & Home Visit",
      //   description:
      //     "Arrange home visit and involve parents in monthly progress reviews.",
      //   expected_impact: "improve family support and reduce risk by ~5 pts",
      //   owner: "Homeroom Teacher + Counsellor"
      // },
      // {
      //   category: "monitoring",
      //   urgency: "routine",
      //   title: "Monthly Risk Review",
      //   description:
      //     "Track attendance and academic performance monthly and rerun simulations when risk factors change.",
      //   expected_impact: "early detection of risk escalation",
      //   owner: "Homeroom Teacher"
      // }
    ],

    narrative:
      "At the ongoing trajectory, Muhammad Ali is at high risk of dropping out within 3 months. Simulated interventions could reduce the risk score by 7 points."
  };

  updateRecommendations(mockCloudResult);
}

function generateMockRecommendations(inputs, score) {
  return [
    {
      category: "attendance",
      urgency: inputs.attendance < 75 ? "critical" : "routine",
      title: inputs.attendance < 75
        ? "Attendance Improvement Plan (AIP)"
        : "Maintain Attendance Monitoring",
      description: inputs.attendance < 75
        ? "Attendance is below threshold. Create a weekly attendance improvement plan."
        : "Attendance is improving. Continue regular monitoring.",
      expected_impact: inputs.attendance < 75
        ? "reduce risk by ~7–10 pts"
        : "early detection of attendance decline"
    },
    {
      category: "academic",
      urgency: inputs.grades < 55 ? "high" : "medium",
      title: inputs.grades < 55
        ? "Peer Tutoring / Additional Support"
        : "Supplementary Practice",
      description: inputs.grades < 55
        ? "Academic score is below benchmark. Assign tutoring support."
        : "Academic performance is acceptable. Provide weekly practice tasks.",
      expected_impact: inputs.grades < 55
        ? "reduce risk score by ~5–8 pts"
        : "prevent academic risk from increasing"
    },
    {
      category: "counselling",
      urgency: inputs.counselling < 3 ? "high" : "routine",
      title: inputs.counselling < 3
        ? "Begin Structured Counselling Programme"
        : "Continue Counselling Follow-Up",
      description: inputs.counselling < 3
        ? "Schedule at least 3 counselling sessions to support engagement."
        : "Counselling support has been planned. Continue follow-up.",
      expected_impact: inputs.counselling < 3
        ? "reduce risk score by ~8 pts"
        : "stabilise student motivation"
    }
  ];
}

function getRecommendationIcon(category) {
  const text = String(category || "").toLowerCase();

  if (text.includes("attendance")) return "📅";
  if (text.includes("academic")) return "🎓";
  if (text.includes("counselling")) return "🤝";
  if (text.includes("welfare")) return "💰";
  if (text.includes("family")) return "🏠";
  if (text.includes("monitoring")) return "🔍";

  return "•";
}

function getRiskColor(score) {
  if (score <= 35) return "#59c56b";
  if (score <= 65) return "#d9de55";
  if (score <= 80) return "#f2a13a";
  return "#e5332f";
}

function resetSimulation() {
  document.getElementById("attendance").value = defaultInputs.attendance;
  document.getElementById("grades").value = defaultInputs.grades;
  document.getElementById("counselling").value = defaultInputs.counselling;
  document.getElementById("welfare").value = defaultInputs.welfare;

  updateSliderLabels(defaultInputs);

  const defaultResult = generateDefaultResult();
  updateSimulationUI(defaultResult);
  updateRecommendations(defaultResult);
}

function generateMockSimulationResult(inputs) {
  const baselineScore = currentStudent.currentScore;

  const isDefault =
    inputs.attendance === defaultInputs.attendance &&
    inputs.grades === defaultInputs.grades &&
    inputs.counselling === defaultInputs.counselling &&
    inputs.welfare === defaultInputs.welfare;

  let score = baselineScore;

  if (!isDefault) {
    score -= (inputs.attendance - defaultInputs.attendance) * 0.25;
    score -= (inputs.grades - defaultInputs.grades) * 0.2;
    score -= inputs.counselling * 1.2;
    score -= inputs.welfare * 3;
  }

  score = Math.round(Math.max(1, Math.min(100, score)));

  const change = baselineScore - score;

  let scoreChangeText = "No change";

  if (change > 0) {
    scoreChangeText = `↓ ${change} pts`;
  } else if (change < 0) {
    scoreChangeText = `↑ ${Math.abs(change)} pts`;
  }

  let riskLevel;
  let riskLabel;

  if (score < 40) {
    riskLevel = "Low Dropout Risk";
    riskLabel = "low risk";
  } else if (score < 65) {
    riskLevel = "Moderate Dropout Risk";
    riskLabel = "medium risk";
  } else {
    riskLevel = "High Dropout Risk";
    riskLabel = "high risk";
  }

  return {
    baselineScore,
    projectedScore: score,
    riskLevel,
    riskLabel,
    scoreChangeText,
    dropoutProbability: score >= 70 ? 90 : score >= 40 ? 55 : 20,
    narrative:
      "At the ongoing trajectory, Muhammad Ali is at high risk of dropping out within 3 months. Immediate action is recommended.",
    insightText:
      change > 0
        ? `Simulation indicates a ${change} point improvement compared to the current baseline if interventions are applied.`
        : change < 0
        ? `Simulation indicates a ${Math.abs(change)} point increase in risk compared to the current baseline.`
        : "Simulation indicates no major improvement compared to the current baseline.",
    weights: {
      attendance: 45,
      academic: 29,
      socioeconomic: 13,
      family: 12
    },
    recommendations: generateMockRecommendations(inputs, score)
  };
}

function generateDefaultResult() {
  return {
    baselineScore: currentStudent.currentScore,
    projectedScore: currentStudent.currentScore,
    riskLevel: `${currentStudent.currentRisk.replace("Risk", "Dropout Risk")}`,
    riskLabel: currentStudent.currentRisk.toLowerCase(),
    scoreChangeText: "No change",

    dropoutProbability: "-",
    probabilityText:
      "No likelihood of dropping out available. Run a simulation to generate dropout probability.",

    insightText:
      "No simulation has been applied yet. Adjust the sliders to generate a projected risk score.",

    weights: {
      attendance: 0,
      academic: 0,
      socioeconomic: 0,
      family: 0
    },

    recommendations: []
  };
}
