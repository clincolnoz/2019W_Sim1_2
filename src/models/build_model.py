from imageai.Prediction.Custom import ModelTraining


def main():
    model_trainer = ModelTraining()
    model_trainer.setModelTypeAsResNet()
    model_trainer.setDataDirectory("..\\data\\images")
    model_trainer.trainModel(num_objects=2,
                             num_experiments=1,
                             enhance_data=True,
                             batch_size=32,
                             show_network_summary=True)


if __name__ == '__main__':
    main()
