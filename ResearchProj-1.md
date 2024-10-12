两次头脑风暴，确定了科研项目的大方向，小兴奋。。。

记录一下待提前研究的问题。
1. 项目中需要用到统计学相关知识，这两天认真在看**假设检验**的知识，花了不少时间理解显著性水平significant level和置信水平confidence level，终于理解了。
   好的解释总是直接击中痛点。
   
   <img width="413" alt="image" src="https://github.com/MaxGYX/Road2Next/assets/158791943/fb4bb75a-6be0-47a2-9081-9f3a6132725d">


A lot of new knowledge to learn, a little busy.


**2024.07.22**

今天老师说到对AI项目来说，数据集至关重要，才意识到这部分的工作量是很大的，且很多需要人工完成。
一个看起来很酷炫的项目背后，首先是很苦逼的数据收集和处理。。。


**sonoteller**，一个分析歌曲流派，风格特征的网站

https://sonoteller.ai/

**Suno Chirp V1关键词**

https://mrkwtkr.notion.site/d6ed0dc92153450c8e7c8c15639eb6ef?v=23aff0603ebf4477bde2b598c8c32974

### 音乐一些基本概念
-   **音乐**是由一些“乐音”按照一定的相互关系组织在一起的声音艺术。
-   **乐音**就是音高固定而且每个音都是**半音的整数倍**高度的“一堆”音。
-   **音程**（Interval）是两个乐音之间的音高关系，通过“度”数确定。“度”用来衡量音与音之间的“距离”。
-   **半音**是指两个音高度上的距离，他是音乐中最小的距离单位。
-   乐音就是从低往高一个比一个高半音的半音阶，共有88个，每个的振动频率都是固定的，最低的音频率为27.5Hz，最高的为4186.0076Hz。这88个音就是乐音，它构成“乐音体系”
-   **旋律**：乐音随时间的延续、变化，构成旋律（也叫曲调）。旋律是音高的时间构成。旋律是由特定的音高变化规律和节奏关系组织起来的一系列音符，其实就是一串乐音的组合。
-   **和弦**（chord）：任何不止一种音高的音同时发生，形成和声（harmony），三个或三个以上不同音高的音，按照三度音程重叠，构成和弦。
-   旋律是歌曲的灵魂，和弦是歌曲的血肉。

-   **声谱**（spectrum of sound）
     -   描述声音的要素：音长（时值），音高（频率），响度（音强），音色（实际上音色由前三个要素构成）
          -   物理学描述声音：一个声音持续0.5ms（音长），频率440Hz（音高），声强级是60dB（响度）。
          -   **音色：** 声音的品质叫做音色（也叫音品），它反映了每个物体发出的声音特有的品质。音色主要由泛音决定。每个人的声音以及各种乐器所发出的声音的区别，就是由音色不同造成的。
     -   **基音**：物体（例如琴弦）在振动时，它的整体振动能产生一个频率最低、强度相对最大的声音，这个声音称为基音。因为基音的音量最大，人耳最容易听见，所以它是音高的决定性因素。
     -   **泛音**：物体（例如琴弦）在振动时，还能产生基音频率2倍、3倍……等频率的振动，这是由驻波的特性决定的，这些频率产生的声音称为泛音，它们的相对音量比较小，不易被人耳听见。
     -   **复合音**：基音与泛音的叠加称为复合音，事实上我们平时听到的声音都是复合音。**正是因为各种声音中泛音的强度不一样，造成了各种音色的区别**。
     -   乐器弹奏一个音符都是由一系列复杂的随时间变化的频率复合而成。音色由泛音决定。
          -   弹下一个琴键的时候，不只是这个音在按照基频在振动，它的2倍、3倍……频率也在同时振动。
     -   下图是两个不同乐器演奏的同一个音高产生的频谱图。可以看出不同琴的音色直观的由不同倍频所决定。
          <img width="563" alt="image" src="https://github.com/user-attachments/assets/be11c833-6b82-4624-b2ce-38c7e42ebde5">

### MIDI （Musical Instrument Digital Interface，音乐设备数字接口）
-   MIDI文件中存储的不是音频数据，而是乐器的演奏指令（wav/mp3这些存储的都是音频数据，而MIDI类似乐谱）。
-   MIDI文件格式
     https://majicdesigns.github.io/MD_MIDIFile/page_smf_definition.html
-   python第三方库解析&生成MIDI文件
      -   pretty_midi https://craffel.github.io/pretty-midi/
      -   mido https://mido.readthedocs.io/en/stable/index.html
   ```python
         import pretty_midi

         midi_data = pretty_midi.PrettyMIDI('hktk.mid')
         for instrument in midi_data.instruments:
             print(instrument.program, instrument.name)
             for note in instrument.notes:
                 print(note)
   ```

### 音乐生成
-   根据MIDI文件对乐器演奏指令的学习，来生成新的音乐是可行的
-   把一首音乐提取频谱图，当成是一副图片。学习图片生成的模型就很多了，但是对于音乐频谱图可能需要一些特别的小手段。

### 频谱图
梅尔频谱图是一种用于表示音频信号频率特征的图像。
-   时间轴（x轴）：
      **表示音频信号的时间进程**，通常以帧（时间窗口）为单位。
      每一列对应于一个时间帧，显示在该时间点的频谱特征。
-   梅尔频率轴（y轴）：
      表示频率的梅尔标度，**梅尔标度是基于人耳对频率的感知**。
      每一行对应于一个梅尔频率带，显示在该频率下的能量值。
-   幅值（颜色强度）：
   通常用颜色或灰度值表示频谱的幅度（能量值）。
   较高的能量值可能显示为较亮的颜色或较深的灰度。

<img width="963" alt="image" src="https://github.com/user-attachments/assets/60a67191-f514-4efd-a200-c4a207dd8891">


### VAE
https://github.com/MaxGYX/Road2Next/blob/main/Learning%20AI%20-%2018%20-%20VAE.md

### Latent Diffusion Model（LDM）
https://github.com/MaxGYX/Road2Next/blob/main/Learning%20AI%20-%2019%20-%20Diffusion.md

### AudioLDM，一个同样vae+ldm，支持condition的音频生成模型，太赞了！
https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm2
    
   

