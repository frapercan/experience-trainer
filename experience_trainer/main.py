import yaml
from experience_trainer.models.actor_critic_resnet import ActorCriticResnet
from experience_trainer.models.imitation_learning_efficientnet import ActorCriticEfficientNet

from experience_trainer.trainning import ExperienceTrainer

if __name__ == "__main__":
    with open("config.yaml", 'r', encoding='utf-8') as conf:
        config = yaml.safe_load(conf)
    handler = ExperienceTrainer(config)
    handler.train_model(ActorCriticEfficientNet, metadata=handler.metadata, epochs=config['epochs'])
