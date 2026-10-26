# تثبيت خادم العرض الوهمي
!apt-get update && apt-get install -y xvfb

# تثبيت المكتبات المعتمدة (النسخ الحديثة تدعم gymnasium)
!pip install stable-baselines3[extra] gymnasium[mujoco] pyvirtualdisplay moviepy


import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# 1. إعداد مجلدات السجلات والحفظ الاحتياطي (للرسم البياني لاحقاً)
log_dir = "./pusher_td3_logs/"
checkpoint_dir = "./pusher_checkpoints/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# 2. إنشاء البيئة وتغليفها بالمراقب
env = gym.make("Pusher-v5")
env = Monitor(env, log_dir)

# 3. إعداد "درع الحماية" (حفظ نسخة من الدماغ كل 50,000 خطوة)
checkpoint_callback = CheckpointCallback(
    save_freq=50000, 
    save_path=checkpoint_dir,
    name_prefix='td3_pusher_backup'
)

# 4. إضافة "ضوضاء الاستكشاف" (السر الخاص بخوارزمية TD3)
# نكتشف عدد المحركات (7 مفاصل) ونضيف لكل منها ضوضاء بنسبة 10%
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

print("🚀 بدء تدريب الذراع الصناعية (Pusher-v4) باستخدام خوارزمية TD3...")

# 5. بناء النموذج بالمعاملات المعتمدة لـ TD3
model = TD3(
    "MlpPolicy",
    env,
    learning_rate=0.001,           # سرعة التعلم القياسية لـ TD3
    buffer_size=1000000,           # ذاكرة عملاقة
    learning_starts=10000,         # جمع 10 آلاف حركة عشوائية قبل التعلم
    batch_size=100,                # حجم حزمة المذاكرة
    tau=0.005,                     # التحديث الناعم لشبكات النقد (Twin Critics)
    gamma=0.99,                    # النظرة المستقبلية
    action_noise=action_noise,     # تمرير الضوضاء التي صنعناها
    verbose=1
)

# 6. بدء التدريب لـ 300 ألف خطوة (مرحلة أولى ممتازة للتقييم)
model.learn(total_timesteps=300_000, log_interval=10, callback=checkpoint_callback)

# 7. الحفظ النهائي 
model.save("td3_pusher_part1")
env.close()

print("✅ تم إكمال المرحلة الأولى بنجاح! تم حفظ دماغ الذراع الصناعية.")



# ==========================================
# تصوير الفيديو للتقييم بعد التطور
# ==========================================

import os
import imageio
from IPython.display import Video, display
from pyvirtualdisplay import Display
from stable_baselines3 import TD3
import gymnasium as gym

# 1. تشغيل الشاشة الوهمية
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

# 2. إنشاء البيئة (إذا استخدمت Pusher-v4 بدلاً من v5 في التدريب، قم بتعديل الاسم هنا)
env_name = "Pusher-v5" 
env_test = gym.make(env_name, render_mode="rgb_array")

# استخراج سرعة الإطارات الصحيحة تلقائياً من البيئة (عادة 20 لبيئة Pusher)
fps = env_test.metadata.get("render_fps", 20)

# 3. تحميل الدماغ الذي تم تدريبه بـ TD3
model_test = TD3.load("td3_pusher_part1")

frames = []
state, _ = env_test.reset()
steps = 0

print(f"🎥 بدء تسجيل ذراع {env_name} (سنسجل 3 محاولات متتالية)...")

# 4. حلقة التصوير لـ 300 خطوة (أي 3 محاولات كاملة)
while steps < 300:
    frames.append(env_test.render())
    
    # القرار الحتمي الدقيق
    action, _states = model_test.predict(state, deterministic=True)
    state, reward, terminated, truncated, _ = env_test.step(action)
    
    # إذا انتهت الـ 100 خطوة، أعد ضبط البيئة فوراً وابدأ المحاولة التالية
    if terminated or truncated:
        state, _ = env_test.reset()
        
    steps += 1

env_test.close()

# 5. حفظ وعرض الفيديو
video_path = './mujoco_pusher_td3.mp4'
imageio.mimsave(video_path, frames, fps=fps, macro_block_size=None)
print(f"✅ تم تسجيل الفيديو بنجاح! إجمالي الإطارات: {len(frames)}")

display(Video(video_path, embed=True))