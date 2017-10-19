#include <jni.h>
#include <string.h>
#include <assert.h>


#if 0
class TestClassProxy : public Observer {
 public:
  JavaObservableProxy(struct HObservable* javaObj, Observable* obs) {
    javaObj_ = javaObj;
    observedOne_ = obs;
    observedOne_->addObserver(this);
  }

  ~JavaObservableProxy() {
    observedOne_->deleteObserver(this);
  }

  void update()  {
    execute_java_dynamic_method(0, javaObj_, "start","()V");
  }

 private:
  struct HObservable* javaObj_;
  Observable* observedOne_;
};
#endif

int main()
{
  JavaVM *jvm;       /* denotes a Java VM */
  JNIEnv *env;       /* pointer to native method interface */
  JavaVMInitArgs vm_args; /* JDK/JRE 6 VM initialization arguments */  
  JavaVMOption options[3];
  
  options[0].optionString = "-Djava.class.path=/home/danielwaddington/comanche/sandpit/jni/javaobj/JavaObj/dist/JavaObj.jar";
  options[1].optionString = "-Djava.library.path=/home/danielwaddington/comanche/sandpit/jni/javaobj/JavaObj/dist"; 
  options[2].optionString = "-verbose:jni";
  
  vm_args.version = JNI_VERSION_1_2;
  vm_args.nOptions = 2;
  vm_args.options = options;
  vm_args.ignoreUnrecognized = JNI_FALSE;
  /* load and initialize a Java VM, return a JNI interface
   * pointer in env */
  JNI_CreateJavaVM(&jvm, (void**) &env, &vm_args);

  /* invoke the Main.test method using the JNI */
  jclass cls = env->FindClass("TestClass");

  if(cls == nullptr)
    printf("error: class not found\n");
  else
    printf("class found Ok\n");
  
  jmethodID mid = env->GetStaticMethodID(cls, "do_something", "()V");
  assert(mid!=0);
  printf("method found Ok mid=%lu\n",(unsigned long)mid);
  
  env->CallStaticVoidMethod(cls, mid);

  printf("method called OK!");
  
  /* We are done. */
  jvm->DestroyJavaVM();

  
   
  return 0;
}
