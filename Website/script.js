/* =========================================================
   NAVIGATION
========================================================= */
function scrollToSection(id) {
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: "smooth" });
}

/* =========================================================
   SUMMARY + CHARTS
========================================================= */
fetch("../severity_output/mask_img.json")
  .then(res => res.json())
  .then(data => {
    document.getElementById("severityScore").textContent = data.severity_level;
    document.getElementById("damagedArea").textContent = data.spread_percentage + "%";

    const badge = document.getElementById("severityBadge");
    badge.textContent =
      data.severity_level <= 2 ? "Severity: Low" :
      data.severity_level <= 4 ? "Severity: Medium" :
      "Severity: High";

    new Chart(document.getElementById("severityChart"), {
      type: "bar",
      data: {
        labels: ["Severity"],
        datasets: [{
          data: [data.severity_level*20],
          backgroundColor: "#ef4444",
          barThickness: 22
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: "y",
        scales: {
          x: { min: 0, max: 100 },
          y: { display: false }
        },
        plugins: { legend: { display: false } }
      }
    });

    new Chart(document.getElementById("damageChart"), {
      type: "doughnut",
      data: {
        labels: ["Damaged", "Undamaged"],
        datasets: [{
          data: [data.spread_percentage, 100 - data.spread_percentage],
          backgroundColor: ["#ef4444", "#22c55e"]
        }]
      }
    });
  });

/* =========================================================
   LLM INSIGHTS - FIXED FOR ACTUAL GEMINI FORMAT
========================================================= */
fetch("<your llm server url>", { cache: "no-store" })
  .then(res => res.json())
  .then(data => {
    console.log("LLM Response:", data.text);

    const keyList = document.getElementById("keyInsights");
    const sevList = document.getElementById("severityInsights");

    keyList.innerHTML = "";
    sevList.innerHTML = "";

    if (!data.text) {
      keyList.innerHTML = "<li>No insights available</li>";
      return;
    }

    // CRITICAL: Replace literal \n with actual newlines
    let text = data.text.replace(/\\n/g, "\n");
    
    // Split by the two main sections
    const sections = text.split(/\*\*Severity Interpretation\*\*/i);
    
    let keySection = sections[0] || "";
    let sevSection = sections[1] || "";

    // Remove "**Key Observations**" header from first section
    keySection = keySection.replace(/\*\*Key Observations\*\*/i, "");

    // Function to extract bullet points
    function extractBullets(sectionText) {
      return sectionText
        .split("\n")
        .map(line => line.trim())
        .filter(line => line.startsWith("*"))  // Only lines that start with *
        .map(line => line.replace(/^\*\s*/, "").trim())  // Remove the * and trim
        .filter(line => line.length > 0);  // Remove empty lines
    }

    // Extract and display Key Observations
    const keyPoints = extractBullets(keySection);
    keyPoints.forEach(point => {
      const li = document.createElement("li");
      li.textContent = point;
      keyList.appendChild(li);
    });

    // Extract and display Severity Interpretation
    const sevPoints = extractBullets(sevSection);
    sevPoints.forEach(point => {
      const li = document.createElement("li");
      li.textContent = point;
      sevList.appendChild(li);
    });

    // Fallback if nothing was parsed
    if (keyList.children.length === 0) {
      keyList.innerHTML = "<li>Unable to parse insights. Check console for details.</li>";
      console.error("Failed to parse key observations from:", keySection);
    }
    
    if (sevList.children.length === 0) {
      sevList.innerHTML = "<li>Unable to parse severity interpretation.</li>";
      console.error("Failed to parse severity section from:", sevSection);
    }

    console.log(`Parsed: ${keyPoints.length} key observations, ${sevPoints.length} severity points`);
  })
  .catch(err => {
    console.error("LLM error:", err);
    document.getElementById("keyInsights").innerHTML =
      "<li>Unable to connect to LLM server. Please ensure it's running.</li>";
    document.getElementById("severityInsights").innerHTML =
      "<li>LLM service unavailable</li>";
  });
