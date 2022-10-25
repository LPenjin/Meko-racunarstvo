class DomainElement:

    def __init__(self, *values):
        self.values = list(values)

    def getNumberOfComponents(self) -> int:
        pass

    def getComponentValue(self) -> int:
        pass

    def hashCode(self) -> int:
        pass

    def equals(self, other) -> bool:
        pass

    def __str__(self):
        pass

    @staticmethod
    def of(*args: int):
        pass

class Domain:

    def __init__(self):
        pass

    @staticmethod
    def intRange(min: int, max: int):
        pass

    @staticmethod
    def combine(domain1, domain2):
        pass

    def indexOfElement(self, DomainElement):
        pass

    def elementForIndex(self, index: int):
        pass

class SimpleDomain:

    def __init__(self, first: int, last: int):
        self.first = first
        self.last = last

    def getCardinality(self) -> int:
        pass

    def getComponent(self, index: int):
        pass

    def getNumberOfComponents(self) -> int:
        pass

    def iterator(self):
        pass

    def getFirst(self) -> int:
        pass

    def getLast(self) -> int:
        pass

class CompositeDomain:

    def __init__(self, *args):
        self.SimpleDomain = list(args)

    def iterator(self):
        pass

    def getCardinality(self):
        pass

    def getComponent(self, index: int):
        pass

    def getNumberOfComponents(self) -> int:
        pass

class CalculatedFuzzySet:

    def __init__(self, IDomain, IInitUnaryFunction):
        self.IDomain = IDomain
        self.IIinitUnaryFunction = IInitUnaryFunction

    def getDomain(self):
        pass

    def getValueAt(self, DomainElement):
        pass

class MutableFuzzySet:

    memberships = 0.0

    def __init__(self, IDomain):
        self.IDomain = IDomain

    def getDomain(self):
        pass

    def getValueAt(self, DomainElement):
        pass

    def set(self, DomainElement, double):
        pass

class StandardFuzzySets():

    def __init__(self):
        pass

    @staticmethod
    def IFuntion(a, b):
        pass

    @staticmethod
    def gammaFunction(a, b):
        pass

    @staticmethod
    def lambdaFunction(a, b, c):
        pass

class Operations:

    def __init__(self):
        pass

    @staticmethod
    def unaryOperation(IFuzzySet, IUnaryFunction):
        pass

    @staticmethod
    def binaryOperation(IFuzzySet1, IFuzzySet2, IBinaryFunction):
        pass

    @staticmethod
    def zadehNot():
        pass

    @staticmethod
    def zadehAnd():
        pass

    @staticmethod
    def zadehOr():
        pass

    @staticmethod
    def hamacherTNorm(double: float):
        pass

    @staticmethod
    def hamacherSNorm(double: float):
        pass


if __name__ == "__main__":
    pass

