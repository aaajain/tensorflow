const tf = require('@tensorflow/tfjs');

class AI {
	compile() {
		const model = tf.sequential();
		
		//input layer
		model.add(tf.layers.dense({
			units:3,
			inputShape: [3]
		}));

		//output layer
		model.add(tf.layers.dense({
			units: 2
		}));

		model.compile({
			loss: 'meanSquaredError',
			optimizer: 'sgd'
		});

		return model;			
	}
	run() {
		const model = this.compile();
		
		//input layer
		const xs = tf.tensor2d([
			[0.1, 0.2, 0.3],
			[0.1, 1.0, 0.3],
			[1.0, 1.0, 1.0]
		]);

		//output layer
		const ys = tf.tensor2d([
			[1, 0],
			[0, 1],
			[1, 1]
		]);

		model.fit(xs, ys, {
			epochs:10000
		}).then(()=>{
			const data = tf.tensor2d([
				[1.0, 1.0, 1.0]
			]);
			const prediction = model.predict(data);
			prediction.print();	
		});
	}
}

const ai = new AI();
ai.run();