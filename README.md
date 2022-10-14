# MoeGoe generating-process workflow

**origined from [CjangCjengh/MoeGoe](https://github.com/CjangCjengh/MoeGoe)**

## usage

- install requirements in `requirements.txt`
- prepare the argument list (if choosing `t` mode)
- edit `config.yml`
- run `MoeGoe.py`

## to prepare argument list

---

*a simple example:*

```
#会员制餐厅 [4]
#对话

本餐厅就是为了那些对各种美味食物已经感到厌倦了的人们 [1]
为他们提供与之身份相应的食物
 [2]
欢迎光临~
欢迎光临~ 2 [3]
```

---

[1]: use a whole line to express one sentence

[2]: you are able to use a blank line to make the list easier to read

[3]: use `[blank]2(a number)` to choose the model this sentence uses (default `[blank]1`)

[4]: use `#` at the start of a line to create an annotation line

**it must be a plain text file.**

## about `config.yml`

```yml
#please make sure the number of your models before you submit
config_path: (your full path of config file)
model_path: (your full path of model file)
##########
## choose one of following options
argument_path: (your full path of argument list)
argument_lang: (the language you want to choose) #pay attention to supported languages of your model!!!
##[ZH] [JA] [KO] [SA] [EN] avaliable
#OR
audio_dir: ###audio_directory: audio input must be in form of ms-wav(*.wav)
origin: ###Original speaker ID
target: ###Target speaker ID
##########
output_path: F:\Desktop\haochi\ #(full path)
method: #t for text2voice while v for voice2voice
time_gap: #time gap between each process (seconds)
```

## useful links

- [Pretrained models](https://github.com/CjangCjengh/TTSModels)

***thanks the original developer***

---

<sub>feel free to pull PR or write issues</sub>