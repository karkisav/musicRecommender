from predict import Predictor
p = Predictor("personality_model.pkl")
print(p.feature_names)
# compare against what _answers_to_features generates
answers = [4,5,2,3,3,4,4,2,3,2]
X = p._answers_to_features(answers)
print(X.shape)  # should be (1, 15)