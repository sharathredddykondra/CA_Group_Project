from model.randomforest import RandomForest, ChainedRandomForest


def model_predict(data, df, name):
    results = []
    print("RandomForest")
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)


def model_evaluate(model, data):
    model.print_results(data)


'''Design Choice 1: Chained Multi-Output RandomForest
Author: Rizvi Abbas 
model_predict_chained is the modelling orchestrator for Design Choice 1.
It instantiates ChainedRandomForest using the shared embedding matrix from
the ChainedData object, then calls train(), predict(), and print_results()
in sequence. Its signature mirrors model_predict() so the controller in
main.py requires no knowledge of how the chained model differs internally.
'''

def model_predict_chained(data, df, name):
    print("\nChained Multi-Output RandomForest — Design Choice 1")
    model = ChainedRandomForest(
        "ChainedRandomForest",
        data.embeddings,
        data.y_train_l1
    )
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)
