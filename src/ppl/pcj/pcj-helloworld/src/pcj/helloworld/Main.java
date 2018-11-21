/*
 * Copyright 2018 danielwaddington.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package pcj.helloworld;

import lib.util.persistent.ObjectDirectory;
import lib.util.persistent.PersistentString;

/**
 *
 * @author danielwaddington
 */
public class Main {
    
    public static void main(String[] args) {
        PersistentString x = ObjectDirectory.get("myName", PersistentString.class);
        if(x == null) {
            System.out.println("Object not present: creating new one");
            PersistentString name = new PersistentString("Daniel");
            ObjectDirectory.put("myName", name);
        }
        else {
            System.out.printf("Read existing object:%s\n", x.toString());
        }
        System.out.println("Done.");
    }

}
