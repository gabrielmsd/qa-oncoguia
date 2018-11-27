conf = {
        'n_words': 20000,
        'question_len': 30,
        'answer_len': 30,
        'margin': 0.2,
        'optimizer':'adam',
        
        #'building': {
	'emb_dimension' : 200,
	'modelname': 'oncoqa_ebm100-cnn50-lstm25-maxp',    
	'architecture': [['EMB',200,1],['CNN',50,1],['LSTM',25,1],['OUT',200,1]],
		'freeze': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],       
		'dropout': 0.2,

        #},
        
        #'training': {
            'neg_examples': 5,
	    'batch_size': 100,
            'nb_epoch': 50,
            'validation_split': 0.1,
        #}
    }
