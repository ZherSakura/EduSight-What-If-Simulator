const API_BASE_URL = "http://localhost:8000";

let selectedStudentId = "STU-001";

let currentStudent = {
  id: "STU-001",
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

document.addEventListener("DOMContentLoaded", () => {
  setupSliderListeners();
  setupStudentSearch();

  // Show immediately
  displayStudentProfile(currentStudent);
  updateSimulationUI(generateDefaultResult());
  updateMockRecommendations();

  // Fetch silently in background
  loadStudentData(selectedStudentId);
});

function setupStudentSearch() {
  const searchInput = document.getElementById("studentSearchInput");

  if (!searchInput) return;

  searchInput.addEventListener("keydown", event => {
    if (event.key === "Enter") {
      const studentId = searchInput.value.trim();

      if (!studentId) {
        alert("Please enter a student ID.");
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
      currentScore: data.currentScore || data.current_score || 89,
      currentRisk: data.currentRisk || data.current_risk || "High Risk"
    };

    displayStudentProfile(currentStudent);
  } catch (error) {
    console.warn("Student API unavailable. Using mock student.");

    const mockStudent = getMockStudentById(studentId);

    if (!mockStudent) {
      alert("Student not found.");
      return;
    }

    currentStudent = mockStudent;
    displayStudentProfile(currentStudent);
  }

  defaultInputs = {
    attendance: currentStudent.attendance || 62,
    grades: currentStudent.grades || 60,
    counselling: currentStudent.counselling || 0,
    welfare: currentStudent.welfare || 0
  };

  resetSlidersToStudentDefault();
  updateSimulationUI(generateDefaultResult());
  updateMockRecommendations();
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

function generateDefaultResult() {
  return {
    baselineScore: currentStudent.currentScore,
    projectedScore: currentStudent.currentScore,
    riskLevel: `${currentStudent.currentRisk.replace("Risk", "Dropout Risk")}`,
    riskLabel: currentStudent.currentRisk.toLowerCase(),
    scoreChangeText: "No change",
    dropoutProbability:
      currentStudent.currentScore >= 70 ? 90 :
      currentStudent.currentScore >= 40 ? 55 : 20,
    probabilityText:
      "No simulation has been applied yet. Adjust the sliders to generate a projected result.",
    insightText:
      "No simulation has been applied yet. Adjust the sliders to generate a projected risk score.",
    weights: {
      attendance: 45,
      academic: 29,
      socioeconomic: 13,
      family: 12
    }
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
    `${result.dropoutProbability}%`;

  document.getElementById("probabilityMessage").textContent =
    result.narrative || "";

  document.getElementById("insightText").textContent =
    result.insightText;

  updateWeights(result.weights);
}

function updateWeights(weights) {
  document.getElementById("attendanceWeightText").textContent =
    `${weights.attendance}%`;
  document.getElementById("academicWeightText").textContent =
    `${weights.academic}%`;
  document.getElementById("socioWeightText").textContent =
    `${weights.socioeconomic}%`;
  document.getElementById("familyWeightText").textContent =
    `${weights.family}%`;

  document.getElementById("attendanceWeightBar").style.width =
    `${weights.attendance}%`;
  document.getElementById("academicWeightBar").style.width =
    `${weights.academic}%`;
  document.getElementById("socioWeightBar").style.width =
    `${weights.socioeconomic}%`;
  document.getElementById("familyWeightBar").style.width =
    `${weights.family}%`;
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
    updateMockRecommendations();
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

  runSimulation();
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

  const improvement = baselineScore - score;

  let riskLevel = "High Dropout Risk";
  let riskLabel = "high risk";

  if (score < 40) {
    riskLevel = "Low Dropout Risk";
    riskLabel = "low risk";
  } else if (score < 65) {
    riskLevel = "Moderate Dropout Risk";
    riskLabel = "medium risk";
  }

  return {
    baselineScore,
    projectedScore: score,
    riskLevel,
    riskLabel,
    scoreChangeText: improvement > 0 ? `↓ ${improvement} pts` : "No change",
    dropoutProbability: score >= 70 ? 90 : score >= 40 ? 55 : 20,
    narrative:
      "At the ongoing trajectory, Muhammad Ali is at high risk of dropping out within 3 months. Immediate action is recommended.",
    insightText:
      improvement > 0
        ? `Simulation indicates a ${improvement} point improvement compared to the current baseline if interventions are applied.`
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