

<h1 id="V9U6j">ProLinker-Generator：Design of PROTAC Linker Base on Generation Model Using Transfer and Reinforcement Learning</h1>
<h2 id="R0uTF">ABSTRACT</h2>z
<font style="color:black;background-color:#FFFFFF;">In PROTAC molecules, the design of the linker directly affects the formation efficiency and stability of the target protein–PROTAC–E3 ligase ternary complex, making it a critical factor in determining degradation activity. However, current linker data are limited, and the accessible chemical space remains narrow. The length, conformation, and chemical composition of linkers play a decisive role in drug performance, highlighting the urgent need for innovative linker design. In this study, we propose ProLinker-Generator, a GPT-based model aimed at generating novel and effective linkers. By integrating transfer learning and reinforcement learning, the model expands the chemical space of linkers and optimizes their design. During the transfer learning phase, the model achieved high scores in validity (0.989) and novelty (0.968) for the generated molecules. In the reinforcement learning phase, it further guided the generation of molecules with ideal properties within our predefined range. ProLinker-Generator demonstrates the significant potential of AI in linker design.</font>

<h2 id="M5vDQ">Necessary package</h2>
Recommended installation under Linux

```plain
python = 3.7.16
pytorch = 1.13.1
numpy
pandas
wandb
RDKit = 2020.09.1.0
```

<h2 id="KPvBG">Data and Models</h2>
We provide the data used, the trained ProLinker-Generator pre-trained model, and the fine-tuned model after transfer learning.

Pre-training data can be downloaded from the link below.

[ChEMBL](https://www.ebi.ac.uk/chembl/)

[ZINC](https://zinc12.docking.org/)

[QM9](https://paperswithcode.com/dataset/qm9)

 The Linkers dataset is from [PROTAC-DB]([Error](http://cadd.zju.edu.cn/protacdb/)), which was made public by the group of Tingjun Hou  from Zhejiang University

pre-training models are placed under `model/pretrain` folder

Fine-tuning models are placed under `model/fine-tuning` folder

<h2 id="RiLtE">Getting Started</h2>
Users can customize their own tasks by modifying the code.  Users can run the ProLinker-Generator model by excuting the 1-4 .py files in sequence according to the following script.

`pretraining.py` is used for pre-training models. Pre-training datasets can be replaced by modifying read paths.

```plain
  python 1_pretrain_ProLinker-Generator.py --run_name{name_for_wandb_run} --data_path{your_pretrain_data}
```

`<font style="color:rgb(44, 44, 54);">generation.py</font>`<font style="color:rgb(44, 44, 54);"> is used for molecule generation and save the generated molecules in CSV format.</font>

```plain
python 2_gen_ProLinker-Generator.py --gen_size{number_of_times_to_generate_from_a_batch} --csv_path{save_path_for_generate_moleculars}
```

`fine_tuning.py` is used for transfer learning. Fine-tuning datasets can be replaced by modifying read paths.

```plain
python 3_fine_tuning_ProLinker-Generator.py --run_name{name_for_wandb_run} --data_path{your_fine_tuning_data}
```

`<font style="color:rgb(44, 44, 54);">RL.py</font>`<font style="color:rgb(44, 44, 54);"> is used for reinforcement learning. By default, it uses molecular properties such as SMILES length , and QED.</font>

```plain
python 4_RL_ProLinker-Generator.py --max_epoch{total_epochs} --bach_size 128 --path{path_to_save_agent_model}
```

<h2 id="XdWqS">License</h2>
This project is licensed under the MIT License.

