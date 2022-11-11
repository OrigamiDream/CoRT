document.addEventListener('DOMContentLoaded', () => {
    const sentence = document.getElementById('sentence');
    const submitButton = document.getElementById('submit');
    const predictions = document.getElementById('predictions');
    const labelNames = [
        '문제 정의', '가설 설정', '기술 정의',
        '제안 방법', '대상 데이터', '데이터처리', '이론/모형',
        '성능/효과', '후속연구'
    ]

    function createCardElements(text) {
        const card = document.createElement('div');
        card.classList.add('card');

        const cardBody = document.createElement('div');
        cardBody.classList.add('card-body');

        const row = document.createElement('div');
        row.classList.add('card-query-row');

        const queryWrap = document.createElement('div');
        queryWrap.classList.add('card-query-wrap');

        const queryBody = document.createElement('div');
        queryBody.classList.add('card-query-body');

        const span = document.createElement('span');
        span.innerText = text;

        const cardTag = document.createElement('div');
        cardTag.classList.add('card-tag');

        const spinnerButton = document.createElement('div');
        spinnerButton.classList.add('btn', 'btn-outline-secondary');

        const spinner = document.createElement('div');
        spinner.classList.add('spinner-border', 'spinner-border-sm');
        spinner.setAttribute('role', 'status');

        const spinnerSpan = document.createElement('span');
        spinnerSpan.classList.add('visually-hidden');
        spinnerSpan.innerText = 'Loading...';

        // prediction
        spinner.append(spinnerSpan);
        spinnerButton.append(spinner);
        cardTag.append(spinnerButton);

        // display area
        queryBody.append(span);
        queryWrap.append(queryBody);

        row.append(queryWrap);
        row.append(cardTag);

        cardBody.append(row);
        card.append(cardBody);

        return {
            card: card,
            row: row,
            tag: cardTag,
            body: queryBody,
            wrap: queryWrap,
            span: span,
            spinnerButton: spinnerButton
        }
    }

    function calculateTokenWidths(tokens) {
        const canvas = calculateTokenWidths.canvas || (calculateTokenWidths.canvas = document.createElement('canvas'));
        const ctx = canvas.getContext('2d');
        const widths = [];
        for(const token of tokens) {
            widths.push(ctx.measureText(token).width);
        }
        return widths;
    }

    function calculatePairedMovingAverage(scalars, left = 2, right = 2) {
        scalars = scalars || [];
        if(!scalars || scalars.length < left + right + 1) {
            return scalars;
        }
        const ma = [];
        for(let i = 0; i < scalars.length; i++) {
            let sum = 0;
            let count = 0;
            for(let j = Math.max(0, i - left); j < Math.min(scalars.length - 1, i + right); j++) {
                sum += scalars[j];
                count++;
            }
            const avg = sum / count;
            ma.push(avg);
        }
        return ma;
    }

    function createAttentionChart(elements, composedTokens) {
        const chartWrap = document.createElement('div');
        chartWrap.classList.add('card-pred-chart');

        const canvas = document.createElement('canvas');
        canvas.setAttribute('width', '0');
        canvas.setAttribute('height', '50');
        canvas.style.width = '0px';
        canvas.style.height = '50px';

        chartWrap.append(canvas);
        elements.body.prepend(chartWrap);
        elements.row.classList.add('pred-shown');

        const tokens = [];
        const scores = [];
        for(const token of composedTokens) {
            tokens.push(token.text);
            scores.push(token.score);
        }
        const tokenWidths = calculateTokenWidths(tokens);
        let digitalized = [];
        for(let i = 0; i < tokenWidths.length; i++) {
            const width = tokenWidths[i];
            const score = scores[i] + 0.05;  // bottom padding
            const repeats = (width / 1) | 0;  // double to int
            for(let j = 0; j < repeats; j++) {
                digitalized.push(score);
            }
        }
        for(let i = 0; i < 3; i++) {
            digitalized = calculatePairedMovingAverage(digitalized);
        }
        const labels = [];
        for(let i = 0; i < digitalized.length; i++) {
            labels.push(`i${i}`);
        }

        const context = canvas.getContext('2d');
        const gradient = context.createLinearGradient(0, 0, 0, 50);
        gradient.addColorStop(0, '#629cff');
        gradient.addColorStop(1, '#ffffff');

        const chart = new Chart(context, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    data: digitalized,
                    backgroundColor: gradient,
                    borderColor: '#629cff',
                    fill: true
                }]
            },
            options: {
                response: true,
                maintainAspectRatio: false,
                events: [],
                plugins: {
                    title: {
                        display: false
                    },
                    legend: {
                        display: false
                    }
                },
                elements: {
                    point: {
                        radius: 0
                    }
                },
                scales: {
                    x: {
                        display: false
                    },
                    y: {
                        display: false
                    }
                },
                animation: {
                    onComplete: function() {
                        const sourceCanvas = chart.ctx.canvas;
                        const destination = document.createElement('canvas');

                        destination.style.width = `${elements.span.offsetWidth}px`;
                        destination.style.height = '50px';
                        destination.style.display = 'block';
                        destination.style.boxSizing = 'border-box';

                        const destinationContext = destination.getContext('2d');
                        const newWidth = elements.span.offsetWidth * 2;
                        const newHeight = 100;
                        destinationContext.canvas.width = newWidth;
                        destinationContext.canvas.height = newHeight;
                        destinationContext.drawImage(sourceCanvas, 0, 0, newWidth, newHeight);

                        canvas.remove();
                        chartWrap.prepend(destination);

                        chart.destroy()
                    }
                }
            }
        });
        function update() {
            const width = elements.span.offsetWidth;
            chart.resize(width, 50);
            chart.update();

            canvas.style.width = `${width}px`;
            canvas.style.height = '50px';
        }

        // no way to force chart size explicitly.
        setTimeout(() => {
            update();
        }, 100);
        setTimeout(() => {
            update();
        }, 500);
    }

    function createPredictionButton(elements, predicted, predicted_prob) {
        const tag = elements.tag;
        const resultButton = document.createElement('button');
        resultButton.type = 'button';
        resultButton.classList.add('btn', 'btn-outline-primary');
        resultButton.innerText = labelNames[predicted];
        resultButton.setAttribute('data-toggle', 'tooltip');
        resultButton.setAttribute('data-placement', 'top');
        resultButton.setAttribute('title', `Confidence: ${predicted_prob.toFixed(4)}`);
        tag.append(resultButton);

        new bootstrap.Tooltip(resultButton);

        resultButton.addEventListener('click', () => {
            if(elements.wrap.classList.contains('probs-shown')) {
                elements.wrap.classList.remove('probs-shown');
            } else {
                elements.wrap.classList.add('probs-shown');
            }
        });
    }

    function createConfidenceChart(elements, probs) {
        const probWrap = document.createElement('div');
        probWrap.classList.add('card-query-prob');
        elements.wrap.append(probWrap);

        const canvas = document.createElement('canvas');
        probWrap.append(canvas);

        const context = canvas.getContext('2d');
        const gradient = context.createLinearGradient(0, 0, 0, 70);
        gradient.addColorStop(0, '#629cff');
        gradient.addColorStop(1, '#ffffff');
        new Chart(context, {
            type: 'bar',
            plugins: [ChartDataLabels],
            data: {
                labels: labelNames,
                datasets: [{
                    label: 'Confidence',
                    data: probs,
                    backgroundColor: gradient,
                    borderColor: '#629cff',
                    borderRadius: 7,
                    borderWidth: 3,
                    fill: true
                }]
            },
            options: {
                response: true,
                maintainAspectRatio: false,
                events: [],
                plugins: {
                    title: {
                        display: false
                    },
                    legend: {
                        display: false
                    },
                    datalabels: {
                        borderRadius: 3,
                        anchor: 'end',
                        align: (context) => {
                            const value = context.dataset.data[context.dataIndex];
                            return value > 0.6 ? 'bottom' : 'top';
                        },
                        color: 'rgb(33, 37, 41)',
                        offset: 5,
                        padding: 0,
                        formatter: (value) => {
                            return value.toFixed(4);
                        },
                    }
                },
                scales: {
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: 'rgb(33, 37, 41)',
                        }
                    },
                    y: {
                        min: 0.0,
                        max: 1.0
                    }
                }
            }
        });
    }

    function predict() {
        const value = sentence.value;
        if(!value) {
            return;
        }
        const elements = createCardElements(value);
        predictions.prepend(elements.card);
        sentence.value = '';

        fetch('/predict', {
            method: 'POST',
            body: JSON.stringify({
                'sentence': value
            }),
            headers: {
                'Content-Type': 'application/json'
            }
        }).then((response) => {
            return response.json()
        }).then((res) => {
            const error = res['error'];
            if(error) {
                throw error;
            }

            const predicted = res['prediction'];
            const predicted_prob = res['prediction_prob'];
            const probs = res['probs'];
            const composedTokens = res['composed_tokens'];

            elements.spinnerButton.remove();
            elements.spinnerButton = null;

            createPredictionButton(elements, predicted, predicted_prob);
            createAttentionChart(elements, composedTokens);
            createConfidenceChart(elements, probs);
        }).catch((error) => {
            const failure = document.createElement('i');
            failure.classList.add('bi', 'bi-exclamation-triangle');

            const button = document.createElement('div');
            button.classList.add('btn', 'btn-outline-danger');
            button.append(failure);

            elements.spinnerButton.remove();
            elements.tag.append(button);

            console.error(error);
        });
    }

    submitButton.addEventListener('click', () => {
        predict();
    });

    sentence.addEventListener('keyup', (e) => {
        if(e.key === 'Enter') {
            predict();
        }
    });
});