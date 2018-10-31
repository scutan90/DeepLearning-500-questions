# 自定义滑动开关按钮-SwitchButton-进阶
> Markdown Revision 1;  --update 2018/10/31 10:55
> Date: 2018/10/31  
> Editor: yang  
> Contact: 1101255053@qq.com  

之前写过这样一个简单的控件，详情可见：
[自定义滑动开关按钮-SwitchButton](http://blog.csdn.net/u012424449/article/details/51678311)
发现还是有点不方便，存在一些问题：
<br/>1.字体没有居中显示 
<br/>2.图片和字都是固定的，不灵活 
<br/>于是在这基础上进行了改良，效果如下，和上次效果几乎一样。本来是有点击事件的，但是这录屏软件好像屏蔽了，录制的时候不会出现，录制完了就会出现屏蔽掉的Toast。

![滑动开关](https://raw.githubusercontent.com/yangmyc/DeepLearning-500-questions/master/img/20160625190025527.gif  "开关")
	
	1.首先在values目录里添加xml文件:attrs，自定义属性,具体如下：
	attrs.xml
	

```
<resources>
    <declare-styleable name="AutoButton">
        <attr name="textOn" format="string"/>
        <attr name="textOff" format="string"/>
        <attr name="textSize_ab" format="dimension"/>
        <attr name="bg_bitmap" format="reference"/>
        <attr name="btn_bitmap" format="reference"/>
    </declare-styleable>
</resources>
```
		
	2.新建一个类继承系统View,在创建的时候获取xml中的属性，具体代码如下：
```
 	TypedArray a = context.obtainStyledAttributes(attrs, R.styleable.AutoButton);
        Drawable bg_Drawable = a.getDrawable(R.styleable.AutoButton_bg_bitmap);
        Drawable btn_Drawable = a.getDrawable(R.styleable.AutoButton_btn_bitmap);
        textOn = a.getString(R.styleable.AutoButton_textOn);
        textOff = a.getString(R.styleable.AutoButton_textOff);
        textSize = a.getDimension(R.styleable.AutoButton_textSize_ab, 35);
        //注意此操作
        a.recycle();
        bgBitmap = ((BitmapDrawable) bg_Drawable).getBitmap();
        btnBitmap = ((BitmapDrawable) btn_Drawable).getBitmap();
```

在上面代码我标注了注意的操作，因为TypeArray是一个单例模式，这个 array 是从一个 array pool的池中获取的。具体可以参考这篇文章：http://blog.csdn.net/Monicabg/article/details/45014327

主要也就是这两个步骤，其他的我就直接上源码了:

AuttoButton.java

```
import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;


/**
 * Created by admin on 2016/6/13.
 * author: myc
 * CopyRight:
 * email:myc1101255053@163.com
 * description:
 */
public class AutoButton extends View implements View.OnTouchListener {
    //背景图片
    private Bitmap bgBitmap;
    //按钮图片
    private Bitmap btnBitmap;
    private Paint paint;
    private int leftDis = 0;
    //标记最大滑动
    private int slidingMax;
    //标记按钮开关状态
    private boolean mCurrent;
    //标记是否点击事件
    private boolean isClickable;
    //标记是否移动
    private boolean isMove;
    //"开"事件监听器
    private SoftFloorListener softFloorListener;
    //"关"事件监听器
    private HydropowerListener hydropowerListener;
    //标记开关文本的宽度
    float width1, width2;
    //记录文本中心点 cx1:绘制文本1的x坐标  cx2:绘制文本2的x坐标
    //cy记录绘制文本的高度
    float cx1, cy, cx2;
    //定义"开"文本
    String textOn;
    //定义"关"文本
    String textOff;
    //定义文本大小
    float textSize;

    public AutoButton(Context context) {
        this(context, null);
    }

    public AutoButton(Context context, AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public AutoButton(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        initData(context, attrs);
        initView();
    }
    private void initData(Context context, AttributeSet attrs) {
        TypedArray a = context.obtainStyledAttributes(attrs, R.styleable.AutoButton);
        Drawable bg_Drawable = a.getDrawable(R.styleable.AutoButton_bg_bitmap);
        Drawable btn_Drawable = a.getDrawable(R.styleable.AutoButton_btn_bitmap);
        textOn = a.getString(R.styleable.AutoButton_textOn);
        textOff = a.getString(R.styleable.AutoButton_textOff);
        textSize = a.getDimension(R.styleable.AutoButton_textSize_ab, 35);
        a.recycle();
        bgBitmap = ((BitmapDrawable) bg_Drawable).getBitmap();
        btnBitmap = ((BitmapDrawable) btn_Drawable).getBitmap();
    }

    private void initView() {
        paint = new Paint();
        slidingMax = bgBitmap.getWidth() - btnBitmap.getWidth();
        paint.setTextSize(textSize);
        width1 = paint.measureText(textOn);
        cx1 = btnBitmap.getWidth() / 2 - width1 / 2;

        //测量绘制文本高度
        Paint.FontMetrics fontMetrics=paint.getFontMetrics();
        float fontHeight=fontMetrics.bottom-fontMetrics.top;
        cy = btnBitmap.getHeight() -(btnBitmap.getHeight()-fontHeight)/2-fontMetrics.bottom;
        width2 = paint.measureText(textOff);
        cx2 = (bgBitmap.getWidth() * 2 - btnBitmap.getWidth()) / 2 - width2 / 2;
        paint.setAntiAlias(true);
        setOnTouchListener(this);
    }



    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        setMeasuredDimension(bgBitmap.getWidth(), bgBitmap.getHeight());
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawBitmap(bgBitmap, 0, 0, paint);
        canvas.drawBitmap(btnBitmap, leftDis, 0, paint);
        if (mCurrent) {
            paint.setColor(Color.WHITE);
            canvas.drawText(textOff, cx2, cy, paint);
            paint.setColor(Color.BLACK);
            canvas.drawText(textOn, cx1, cy, paint);
        } else {
            paint.setColor(Color.WHITE);
            canvas.drawText(textOn, cx1, cy, paint);
            paint.setColor(Color.BLACK);
            canvas.drawText(textOff, cx2, cy, paint);
        }


    }


    //刷新视图
    private void flushView() {
        mCurrent = !mCurrent;
        if (mCurrent) {
            leftDis = slidingMax;
            if (hydropowerListener != null) {
                hydropowerListener.hydropower();
            }
        } else {
            leftDis = 0;
            if (softFloorListener != null) {
                softFloorListener.softFloor();
            }
        }
//        System.out.println("mCurrent:="+mCurrent);
        invalidate();
    }

    //startX 标记按下的X坐标,  lastX标记移动后的X坐标 ,disX移动的距离
    float startX, lastX, disX;

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                isClickable = true;
                startX = event.getX();
                isMove = false;
                break;
            case MotionEvent.ACTION_MOVE:
                lastX = event.getX();
                disX = lastX - startX;
                if (Math.abs(disX) < 5) break;
                isMove = true;
                isClickable = false;
                moveBtn();
                startX = event.getX();
                break;
            case MotionEvent.ACTION_UP:
                if (isClickable) {
                    flushView();
                }
                if (isMove) {
                    if (leftDis > slidingMax / 2) {
                        mCurrent = false;
                    } else {
                        mCurrent = true;
                    }
                    flushView();
                }
                break;
        }

        return true;
    }


    //移动后判断位置
    private void moveBtn() {
        leftDis += disX;
        if (leftDis > slidingMax) {
            leftDis = slidingMax;
        } else if (leftDis < 0) {
            leftDis = 0;
        }
        invalidate();
    }


    //设置左边按钮点击事件监听器
    public void setSoftFloorListener(SoftFloorListener softFloorListener) {
        this.softFloorListener = softFloorListener;
    }

    //设置右边按钮点击事件监听器
    public void setHydropowerListener(HydropowerListener hydropowerListener) {
        this.hydropowerListener = hydropowerListener;
    }

    //开点击事件
    public interface SoftFloorListener {
        void softFloor();
    }

    //关点击事件
    public interface HydropowerListener {
        void hydropower();
    }
}

```

然后就是在xml布局里调用，具体如下：


activity_main.xml
```
<RelativeLayout xmlns:android="http://schemas.android.com/apk/res/android"
	<!--注意此步骤-->
    xmlns:myc="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:paddingBottom="@dimen/activity_vertical_margin"
    android:paddingLeft="@dimen/activity_horizontal_margin"
    android:paddingRight="@dimen/activity_horizontal_margin"
    android:paddingTop="@dimen/activity_vertical_margin">

  <com.yigong.admin.autobutton.AutoButton
      android:id="@+id/btn_switchbutton"
      android:layout_width="wrap_content"
      android:layout_height="wrap_content"
      myc:bg_bitmap="@drawable/bg"
      myc:btn_bitmap="@drawable/btn"
      myc:textSize_ab="15sp"
      myc:textOn="开"
      myc:textOff="关"
      android:padding="10dp"
      android:layout_centerHorizontal="true"/>
</RelativeLayout>
```
在上面我标注了注意步骤，因为需要使用自定义的控件属性，所以需要设置xml命名空间
xmlns:myc="http://schemas.android.com/apk/res-auto"，简单点可以这么设置，规范是
xmlns:myc="http://schemas.android.com/apk/<这里是包名>"

MainActivity.java


```
public class MainActivity extends Activity {
    private AutoButton btn_switchbutton;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
       initView();
    }

    private void initView() {
        btn_switchbutton= (AutoButton) findViewById(R.id.btn_switchbutton);
        btn_switchbutton.setHydropowerListener(hydropowerListener);
        btn_switchbutton.setSoftFloorListener(softFloorListener);
    }
    AutoButton.HydropowerListener hydropowerListener=new AutoButton.HydropowerListener() {
        @Override
        public void hydropower() {
            Toast.makeText(MainActivity.this,"关",Toast.LENGTH_SHORT).show();
        }
    };
    AutoButton.SoftFloorListener softFloorListener=new AutoButton.SoftFloorListener() {
        @Override
        public void softFloor() {
      Toast.makeText(MainActivity.this,"开",Toast.LENGTH_SHORT).show();
        }
    };
}
```

完整代码可前往GitHub:https://github.com/yangmyc/SwitchButton
我这里设置了点击事件，因为录屏软件问题，所以没有显示，关了录屏软件后，之前的Toast就全出来了。有问题可以私聊，或者你有更好的方法，请多多指教。
