# JSON Operators

Synalinks introduce operators that works at the JSON schema level allowing to perform composition, factorization and logical operations of Pydantic structures at the class and type level. They are a key concept to understand as they also allow to control the flow of information in the computation graph. Here is a simple cheatsheet to understand how their works in practice with toy examples.

### Concatenate

When you use the operators with non-instanciated types, the result will be a `SymbolicDataModel`.

A symbolic data model is a placeholder for data specification

```python
import synalinks

class Foo(synalinks.DataModel):
    foo: str

class Bar(synalinks.DataModel):
    bar: str

foofoo = Foo + Foo

print(foo.prettify_schema())
```
>>>

```python
foobar = Foo + Bar

print(foobar.prettify_schema())
```
>>>



### Factorize

```python
foofoo = Foo + Foo

foofoo = foofoo.factorize()

print(foo.prettify_schema())
```