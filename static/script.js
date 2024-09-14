document.addEventListener('DOMContentLoaded', function() {
    const ageInput = document.getElementById('age');
    const weightInput = document.getElementById('weight');
    const heightInput = document.getElementById('height');
    const ageValue = document.getElementById('ageValue');
    const weightValue = document.getElementById('weightValue');
    const heightValue = document.getElementById('heightValue');
    const medicalHistorySelect = document.getElementById('medicalHistory');
    const medicalHistoryTags = document.getElementById('medicalHistoryTags');
    const generateScenariosButton = document.getElementById('generateScenarios');
    const scenariosContainer = document.getElementById('scenariosContainer');
    const scenariosDiv = document.getElementById('scenarios');

    const patientData = {
        name: '',
        age: 25,
        weight: 70,
        height: 170,
        medicalHistory: []
    };

    ageInput.addEventListener('input', function() {
        ageValue.textContent = this.value;
        patientData.age = parseInt(this.value);
    });

    weightInput.addEventListener('input', function() {
        weightValue.textContent = this.value;
        patientData.weight = parseInt(this.value);
    });

    heightInput.addEventListener('input', function() {
        heightValue.textContent = this.value;
        patientData.height = parseInt(this.value);
    });

    medicalHistorySelect.addEventListener('change', function() {
        if (this.value && !patientData.medicalHistory.includes(this.value)) {
            patientData.medicalHistory.push(this.value);
            updateMedicalHistoryTags();
        }
        this.value = '';
    });

    function updateMedicalHistoryTags() {
        medicalHistoryTags.innerHTML = '';
        patientData.medicalHistory.forEach(condition => {
            const tag = document.createElement('span');
            tag.className = 'bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded';
            tag.textContent = condition;
            medicalHistoryTags.appendChild(tag);
        });
    }

    generateScenariosButton.addEventListener('click', function() {
        patientData.name = document.getElementById('name').value;
        fetch('/generate_scenarios', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(patientData),
        })
        .then(response => response.json())
        .then(scenarios => {
            displayScenarios(scenarios);
        });
    });

    function displayScenarios(scenarios) {
        scenariosDiv.innerHTML = '';
        scenarios.forEach(scenario => {
            const scenarioElement = document.createElement('div');
            scenarioElement.className = 'bg-white p-4 rounded-lg shadow mb-4';
            scenarioElement.innerHTML = `
                <h3 class="text-lg font-medium">${scenario.name}</h3>
                <p class="text-sm text-gray-500">Probability: ${(scenario.probability * 100).toFixed(1)}%</p>
            `;
            scenariosDiv.appendChild(scenarioElement);
        });
        scenariosContainer.classList.remove('hidden');
    }
});
