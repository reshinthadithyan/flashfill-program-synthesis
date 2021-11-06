import os
import re
from torch.utils.data import Dataset, random_split
from tqdm import tqdm
from transformers import RobertaTokenizerFast
from torch import Generator


class FlashFillDataset(Dataset):
    def __init__(self, dataset_idt, tokenizer, target_length):
        """
        FlashFill torch.utils.data.Dataset object
        args:
            dataset_idt (str) : Dataset Name for identification [FlashFill]
            tokenizer   (transformers.AutoTokenizer)  : Transformers AutoTokenizer for tokenizing the data
            target_length (int) : maximum target length for the model
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset_idt = dataset_idt
        self.target_length = target_length
        self.dataset_list = self.read_and_process()

    def clean_flash_fill_data(self, data):
        """
        Function to clean and pre-process the datapoint suitable for CausalLM
        """
        # TOOD(reshinth) : Clean multiple \n to one
        data = re.sub(r"\n+", "\n", data).strip()
        data_split_list = data.split("\n(declare-var")
        to_generate = data_split_list[0]
        init_context = "\n(declar-var" + "\n(declar-var".join(data_split_list[1:])
        return init_context, to_generate

    def read_and_process(
        self, path=os.path.join(".", "DeepSynth", "flashfill_dataset")
    ):
        """
        read and process files into datapoints
        args:
            path (str) : path for the dataset files.
        """
        dataset_list = []
        if self.dataset_idt == "FlashFill":
            file_names = [
                os.path.join(path, file_name) for file_name in os.listdir(path)
            ]
            for file in tqdm(file_names):
                with open(file, "r") as f:
                    init_context, to_generate = self.clean_flash_fill_data(f.read())
                    inputs = init_context + "<generate-flashfill>" + to_generate
                    model_inputs = self.tokenizer(
                        inputs,
                        max_length=self.target_length,
                        padding="max_length",
                        truncation=True,
                    )
                    labels = self.tokenizer(
                        to_generate,
                        max_length=self.target_length,
                        padding="max_length",
                        truncation=True,
                    ).input_ids
                    labels_example = [label if label != 0 else -100 for label in labels]
                    model_inputs["labels"] = labels_example
                    dataset_list.append(model_inputs)
            return dataset_list
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, index):
        return self.dataset_list[index]


def train_test_valid_split(dataset_idt, tokenizer, max_length):
    flashfill_dataset = FlashFillDataset(dataset_idt, tokenizer, max_length)
    return random_split(
        flashfill_dataset, [92, 8, 8], generator=Generator().manual_seed(42)
    )


if __name__ == "__main__":
    from torch import Generator
    from torch.utils.data import random_split

    trial_tokenizer = RobertaTokenizerFast.from_pretrained("Salesforce/codet5-small")
    flashfill_dataset = FlashFillDataset("FlashFill", trial_tokenizer, 1024)
    dataset_length = len(flashfill_dataset)
    print(
        random_split(
            flashfill_dataset, [92, 8, 8], generator=Generator().manual_seed(42)
        )
    )
    print(flashfill_dataset[-1])
