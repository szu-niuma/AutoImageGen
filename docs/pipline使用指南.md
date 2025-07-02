# Pipeline抽象基类使用指南

## 概述

`BasePipeline` 是一个抽象基类，为图像处理管道提供了统一的框架。它封装了常见的功能，如多线程处理、文件管理、进度跟踪和错误处理。

## 主要特性

### 1. 统一的初始化参数
- `config`: 配置字典
- `out_dir`: 输出目录
- `is_debug`: 调试模式（单线程）
- `max_num`: 最大处理数量限制
- `pipeline_name`: 管道名称
- `image_size`: 图像处理尺寸
- `maintain_aspect_ratio`: 是否保持宽高比

### 2. 自动化文件管理
- 支持处理单个图片、目录或JSON文件
- 自动创建输出目录
- 内置结果缓存机制
- 支持批量处理和结果保存

### 3. 多线程处理框架
- 自动选择最优线程数
- 支持调试模式（单线程）
- 内置进度条和错误统计
- 优雅的异常处理

### 4. 可扩展设计
- 抽象方法 `pipeline()` 需要子类实现
- 可重写的钩子方法用于自定义行为
- 灵活的文件过滤和结果保存机制

## 使用方法

### 1. 创建子类

```python
from auto_image_edit.base_pipeline import BasePipeline

class MyPipeline(BasePipeline):
    def __init__(self, config, **kwargs):
        super().__init__(
            config=config,
            pipeline_name="MyPipeline",
            image_size=(256, 256),
            **kwargs
        )
        # 初始化你的处理器
        self.my_processor = MyProcessor(config)
    
    def pipeline(self, info):
        """
        实现核心处理逻辑
        Args:
            info: 包含image_path等信息的字典
        Returns:
            处理结果字典
        """
        image_path = info["image_path"]
        
        # 加载图像
        src_img, _, _ = self.image_processor.load_image(image_path)
        image_base64 = self.image_processor.get_base64(src_img)
        
        # 执行处理
        result = self.my_processor.process(image_base64)
        
        return result
```

### 2. 使用Pipeline

```python
# 初始化
config = {"model_path": "path/to/model"}
pipeline = MyPipeline(config, out_dir="output", is_debug=False)

# 处理单个图片
result = pipeline.run("path/to/image.jpg")

# 处理目录
result = pipeline.run("path/to/images/")

# 处理JSON文件
result = pipeline.run("path/to/data.json")

# 限制处理数量
pipeline = MyPipeline(config, max_num=100)
result = pipeline.run("path/to/large_dataset/")
```

### 3. 自定义行为

```python
class CustomPipeline(BasePipeline):
    def _should_process_file(self, file_path):
        """自定义文件过滤逻辑"""
        return file_path.suffix.lower() in {'.png', '.jpg'}
    
    def _check_existing_result(self, info):
        """自定义结果缓存检查"""
        # 可以实现更复杂的缓存逻辑
        return super()._check_existing_result(info)
    
    def _save_result(self, info, result):
        """自定义结果保存逻辑"""
        # 可以保存到数据库、云存储等
        super()._save_result(info, result)
```

## 与原有代码的对比

### 原有代码问题
1. **重复代码**: 三个Pipeline有大量相似的初始化、文件处理和多线程代码
2. **维护困难**: 修改公共逻辑需要在多个文件中重复修改
3. **扩展性差**: 添加新的Pipeline需要重新实现基础功能
4. **一致性差**: 不同Pipeline的接口和行为不统一

### 使用基类的优势
1. **代码复用**: 公共逻辑只需实现一次
2. **易于维护**: 修改基类即可影响所有子类
3. **一致性**: 所有Pipeline有统一的接口和行为
4. **易于扩展**: 新Pipeline只需实现核心逻辑
5. **更好的测试**: 可以针对基类编写通用测试

## 迁移指南

### 1. 保持向后兼容
现有的Pipeline可以继续使用，新的基类Pipeline可以逐步替换：

```python
# 旧代码
from auto_image_edit.diff_pipeline import DiffPipeline

# 新代码 - 逐步迁移
from auto_image_edit.refactored_pipelines import RefactoredDiffPipeline as DiffPipeline
```

### 2. 批量迁移
所有Pipeline都可以通过继承BasePipeline来重构，大幅减少代码量。

### 3. 功能增强
基类提供了更多功能，如统计信息、动态配置等：

```python
# 获取统计信息
stats = pipeline.get_stats()

# 动态设置最大线程数
pipeline.set_max_workers(8)
```

## 最佳实践

1. **命名约定**: Pipeline类名以"Pipeline"结尾
2. **配置管理**: 将所有配置项放在config字典中
3. **错误处理**: 在pipeline方法中适当处理异常
4. **日志记录**: 使用logger记录关键信息
5. **性能优化**: 根据数据量选择合适的线程数和批处理大小

## 扩展功能

基类还可以进一步扩展：

```python
class AdvancedBasePipeline(BasePipeline):
    """支持更多高级功能的基类"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics = {}
        self.callbacks = []
    
    def add_callback(self, callback):
        """添加处理回调"""
        self.callbacks.append(callback)
    
    def _execute_pipeline(self, info):
        """带回调和指标的执行"""
        start_time = time.time()
        
        # 执行前回调
        for callback in self.callbacks:
            callback.on_start(info)
        
        result = super()._execute_pipeline(info)
        
        # 记录指标
        self.metrics[info['image_path']] = time.time() - start_time
        
        # 执行后回调
        for callback in self.callbacks:
            callback.on_complete(info, result)
        
        return result
```

这个抽象基类设计提供了一个强大、灵活且易于使用的框架，可以大大简化新Pipeline的开发和现有代码的维护。
