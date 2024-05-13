from transformers import AutoModelForSeq2SeqLM, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch.nn as nn

class PrefixTuningModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        self.prefix_encoder = nn.Embedding(1, config.d_model)

    def get_prompt(self, batch_size):
        prefix_indices = torch.tensor([self.model.config.decoder_start_token_id])
        return self.prefix_encoder(prefix_indices).expand(batch_size, 1, -1)

    def forward(self, input_ids, attention_mask, labels):
        batch_size = input_ids.shape[0]
        prefix = self.get_prompt(batch_size)
        outputs = self.model(
            inputs_embeds=prefix,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        return outputs

config = PretrainedConfig.from_pretrained("t5-base")
model = PrefixTuningModel(config)

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

trainer.train()
