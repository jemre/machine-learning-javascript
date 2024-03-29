import loadCSV from './utilities/load-csv';
import knn from './algorithms/knn';
const tf = require('@tensorflow/tfjs');

const runKNN = () => {

    let {features, labels, testFeatures, testLabels} = loadCSV('src/data-sets/kc_house_data.csv', {
        shuffle: true,
        splitTest: 2,
        dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living'],
        labelColumns: ['price']
    });

    features = tf.tensor(features);
    labels = tf.tensor(labels);

    testFeatures.forEach((testPoint, i) => {

        const result = knn(features, labels, tf.tensor(testPoint), 10);
        const error = (testLabels[i][0] - result) / testLabels[i][0];

        console.log('error: ', error * 100);

    });

};

export default runKNN;


