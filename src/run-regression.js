import loadCSV from './utilities/load-csv';
import LinearRegression from './algorithms/regressions';
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs');

const runRegression = () => {

    let {features, labels, testFeatures, testLabels} = loadCSV('src/data-sets/cars.csv', {
        shuffle: true,
        splitTest: 50,
        dataColumns: ['horsepower'],
        labelColumns: ['mpg']
    });

    const options = {
        iterations: 100,
        learningRate: 0.0001
    };

    const linearRegression = new LinearRegression(features, labels, options);

    linearRegression.train();

    console.log('linearRegression.b: ', linearRegression.b);
    console.log('linearRegression.m: ', linearRegression.m);

};

export default runRegression;
