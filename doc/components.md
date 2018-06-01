#### Component design

To add a component, you can follow the posix_component example:

1. uuid for interfaces:
  ````
  components/api/block_itf.h
  ````

2. uuid for component:
  ```
  src/components/block_posix_component.h
  ```

3. uuid also need to be declared in the 
  ```
  src/components/api/components.h
  ```
