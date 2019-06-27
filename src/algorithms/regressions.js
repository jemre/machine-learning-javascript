const _ = require('lodash');

const defaultOptions = {
    iterations: 1000,
    learningRate: 0.1
};

class LinearRegression {

    constructor(features, labels, options) {

        this.features = features;
        this.labels = labels;
        this.options = Object.assign(defaultOptions, options);

        this.m = 0;
        this.b = 0;

    }

    gradientDescent() {

        const currentGuessesForMPG = this.features.map(row => this.m * row[0] + this.b);

        const bSlope = _.sum(currentGuessesForMPG.map((guess, i) =>
            guess - this.labels[i][0])) * 2 / this.features.length;

        const mSlope = _.sum(currentGuessesForMPG.map((guess, i) =>
            -1 * this.features[i][0] * (this.labels[i][0] - guess))) * 2 / this.features.length;

        this.b = this.b - bSlope * this.options.learningRate;
        this.m = this.m - mSlope * this.options.learningRate;

    }

    train() {

        for(let i = 0; i < this.options.iterations; i++) {
            this.gradientDescent()
        }

    }

}

export default LinearRegression;
