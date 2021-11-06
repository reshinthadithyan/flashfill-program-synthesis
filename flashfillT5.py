import argparse
from torch.utils.data import random_split,DataLoader
from torch import Generator
from transformers import T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup, RobertaTokenizerFast
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import os
from data_utils import train_test_valid_split
parser = argparse.ArgumentParser()

parser.add_argument("run_name",default="codet5-finetune-program-synthesis")
parser.add_argument("dir",default="flash_fill_hash")
parser.add_argument("model_name",default="Salesforce/codet5-small")
parser.add_argument("dataset_idt",default="FlashFill")
parser.add_argument("max_epochs",default=15)
parser.add_argument("warmup_steps",default=1000)
parser.add_argument("target_max_length",default=1024)

args = parser.parse_args()


class CodeT5(pl.LightningModule):
    #Borrowed from the awesome https://github/NielsRogge/Transformers-Tutorials by Niels Rogge
    def __init__(self, lr=5e-5, num_train_epochs=args.max_epochs, warmup_steps=args.warmup_steps):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(args.model_name)
        tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)
        self.trainset,self.validset,self.testset = train_test_valid_split("FlashFill",tokenizer,args.target_max_length)
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=None):     
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
    
    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        return loss
      
    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     

        return loss

    def configure_optimizers(self):
        # create optimizer
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        # create learning rate scheduler
        num_train_optimization_steps = self.hparams.num_train_epochs * len(self.train_dataloader()) #Quick Dirty Fix
        lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self):
        return DataLoader(self.trainset,batch_size=8,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validset,batch_size=4,shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.testset,batch_size=4,shuffle=True)

model = CodeT5()

wandb_logger = WandbLogger(name='args.run_name', project='FlashFill')
early_stop_callback = EarlyStopping(
    monitor='validation_loss',
    patience=3,
    strict=False,
    verbose=False,
    mode='min'
)
lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = Trainer(gpus=1, 
                  default_root_dir=os.path.join(args.dir,args.run_name,"runs"), 
                  logger=wandb_logger, 
                  callbacks=[early_stop_callback, lr_monitor])
trainer.fit(model)
model.model.save_pretrained(os.path.join(args.dir,args.run_name,"model"))