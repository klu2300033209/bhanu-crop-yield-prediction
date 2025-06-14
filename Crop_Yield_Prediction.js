// 1. Define the dataset (50 rows) with features and target
const data = [
    { rainfall: 400, meanTemp: 25, alkaline: 7.2, yield: 2.8 },
    { rainfall: 350, meanTemp: 27, alkaline: 7.0, yield: 2.6 },
    { rainfall: 500, meanTemp: 30, alkaline: 8.0, yield: 3.5 },
    { rainfall: 450, meanTemp: 24, alkaline: 6.8, yield: 2.7 },
    { rainfall: 550, meanTemp: 28, alkaline: 7.5, yield: 3.6 },
    { rainfall: 300, meanTemp: 22, alkaline: 7.1, yield: 2.3 },
    { rainfall: 380, meanTemp: 26, alkaline: 6.9, yield: 2.9 },
    { rainfall: 470, meanTemp: 29, alkaline: 7.3, yield: 3.3 },
    { rainfall: 420, meanTemp: 23, alkaline: 7.0, yield: 2.5 },
    { rainfall: 330, meanTemp: 25, alkaline: 6.7, yield: 2.4 },
    { rainfall: 390, meanTemp: 26, alkaline: 7.1, yield: 2.8 },
    { rainfall: 470, meanTemp: 27, alkaline: 6.9, yield: 3.0 },
    { rainfall: 510, meanTemp: 28, alkaline: 7.6, yield: 3.7 },
    { rainfall: 560, meanTemp: 29, alkaline: 7.4, yield: 3.9 },
    { rainfall: 460, meanTemp: 24, alkaline: 7.2, yield: 2.9 },
    { rainfall: 330, meanTemp: 26, alkaline: 7.0, yield: 2.5 },
    { rainfall: 400, meanTemp: 28, alkaline: 7.3, yield: 3.1 },
    { rainfall: 480, meanTemp: 30, alkaline: 7.5, yield: 3.4 },
    { rainfall: 420, meanTemp: 25, alkaline: 6.8, yield: 2.7 },
    { rainfall: 490, meanTemp: 27, alkaline: 7.2, yield: 3.2 },
    { rainfall: 430, meanTemp: 24, alkaline: 6.9, yield: 2.6 },
    { rainfall: 540, meanTemp: 29, alkaline: 7.6, yield: 3.8 },
    { rainfall: 350, meanTemp: 23, alkaline: 7.0, yield: 2.4 },
    { rainfall: 480, meanTemp: 28, alkaline: 7.4, yield: 3.3 },
    { rainfall: 460, meanTemp: 27, alkaline: 7.3, yield: 3.0 },
    { rainfall: 500, meanTemp: 30, alkaline: 7.5, yield: 3.6 },
    { rainfall: 530, meanTemp: 29, alkaline: 7.2, yield: 3.8 },
    { rainfall: 310, meanTemp: 22, alkaline: 6.8, yield: 2.2 },
    { rainfall: 520, meanTemp: 30, alkaline: 7.6, yield: 3.7 },
    { rainfall: 540, meanTemp: 26, alkaline: 7.4, yield: 3.5 },
    { rainfall: 560, meanTemp: 28, alkaline: 7.3, yield: 3.9 },
    { rainfall: 600, meanTemp: 29, alkaline: 7.7, yield: 4.0 },
    { rainfall: 550, meanTemp: 27, alkaline: 7.2, yield: 3.6 },
    { rainfall: 480, meanTemp: 25, alkaline: 6.9, yield: 2.9 },
    { rainfall: 470, meanTemp: 28, alkaline: 7.4, yield: 3.2 },
    { rainfall: 490, meanTemp: 30, alkaline: 7.1, yield: 3.3 },
    { rainfall: 420, meanTemp: 24, alkaline: 6.8, yield: 2.6 },
    { rainfall: 510, meanTemp: 27, alkaline: 7.5, yield: 3.6 },
    { rainfall: 520, meanTemp: 29, alkaline: 7.3, yield: 3.7 },
    { rainfall: 530, meanTemp: 25, alkaline: 7.2, yield: 3.5 },
    { rainfall: 480, meanTemp: 26, alkaline: 7.0, yield: 3.1 },
    { rainfall: 450, meanTemp: 24, alkaline: 6.9, yield: 2.8 },
    { rainfall: 400, meanTemp: 28, alkaline: 7.4, yield: 3.2 },
    { rainfall: 470, meanTemp: 30, alkaline: 7.1, yield: 3.4 },
    { rainfall: 490, meanTemp: 29, alkaline: 7.3, yield: 3.3 },
    { rainfall: 520, meanTemp: 27, alkaline: 7.6, yield: 3.8 },
    { rainfall: 550, meanTemp: 26, alkaline: 7.2, yield: 3.5 }
];

// 2. Prepare feature matrix (X) and target vector (y)
const X = data.map(d => [1, d.rainfall, d.meanTemp, d.alkaline]); // Add 1 for intercept
const y = data.map(d => d.yield);

// 3. Matrix operations
function matrixTranspose(matrix) {
    return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
}

function matrixMultiply(a, b) {
    const aRows = a.length;
    const aCols = a[0].length;
    const bCols = b[0].length;
    const result = new Array(aRows);
    
    for (let i = 0; i < aRows; i++) {
        result[i] = new Array(bCols).fill(0);
        for (let j = 0; j < bCols; j++) {
            for (let k = 0; k < aCols; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

function matrixInvert(matrix) {
    const n = matrix.length;
    const identity = Array(n).fill().map((_, i) => 
        Array(n).fill().map((_, j) => i === j ? 1 : 0)
    );
    
    const augmented = matrix.map((row, i) => [...row, ...identity[i]]);
    
    for (let i = 0; i < n; i++) {
        let maxRow = i;
        for (let j = i + 1; j < n; j++) {
            if (Math.abs(augmented[j][i]) > Math.abs(augmented[maxRow][i])) {
                maxRow = j;
            }
        }
        
        [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
        const pivot = augmented[i][i];
        
        for (let j = i; j < 2 * n; j++) {
            augmented[i][j] /= pivot;
        }
        
        for (let k = 0; k < n; k++) {
            if (k !== i && augmented[k][i] !== 0) {
                const factor = augmented[k][i];
                for (let j = i; j < 2 * n; j++) {
                    augmented[k][j] -= augmented[i][j] * factor;
                }
            }
        }
    }
    
    return augmented.map(row => row.slice(n));
}

// 4. Perform Multiple Linear Regression
function linearRegression(X, y) {
    const X_T = matrixTranspose(X);
    const X_T_X = matrixMultiply(X_T, X);
    const X_T_X_inv = matrixInvert(X_T_X);
    const X_T_y = matrixMultiply(X_T, y.map(v => [v]));
    const theta = matrixMultiply(X_T_X_inv, X_T_y);
    
    return theta.map(v => v[0]); // Convert to simple array
}

// 5. Train the model and get coefficients
const coefficients = linearRegression(X, y);
console.log("Model coefficients (intercept, rainfall, meanTemp, alkaline):", coefficients);

// 6. Prediction function (exposed to HTML)
function predict(features) {
    if (!coefficients || coefficients.length !== 4) {
        console.error("Model coefficients not properly initialized");
        return 0;
    }
    
    const [intercept, rainfallCoef, tempCoef, alkalineCoef] = coefficients;
    return intercept + 
           rainfallCoef * features[0] + 
           tempCoef * features[1] + 
           alkalineCoef * features[2];
}

// 7. Model evaluation
function calculateRSquared(X, y, coefficients) {
    const yMean = y.reduce((sum, val) => sum + val, 0) / y.length;
    let ssTotal = 0;
    let ssResidual = 0;
    
    for (let i = 0; i < y.length; i++) {
        const prediction = predict([X[i][1], X[i][2], X[i][3]]);
        ssTotal += Math.pow(y[i] - yMean, 2);
        ssResidual += Math.pow(y[i] - prediction, 2);
    }
    
    return 1 - (ssResidual / ssTotal);
}

const rSquared = calculateRSquared(X, y, coefficients);
console.log(`R-squared: ${rSquared.toFixed(4)}`);