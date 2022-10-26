from abc import ABC, abstractmethod
from itertools import product
from numpy import prod


class DomainElement:

    def __init__(self, *values):
        self.values = list(values)

    def getNumberOfComponents(self) -> int:
        return len(self.values)

    def getComponentValue(self) -> int:
        return self.values

    def hashCode(self) -> int:
        pass

    def equals(self, other) -> bool:
        return other.getComponentValue() == self.getComponentValue()

    def __str__(self):
        if len(self.values) == 1:
            return "(" + str(self.values[0]) + ")"
        else:
            return "(" + ",".join([str(element) for element in self.values]) + ")"

    @staticmethod
    def of(*args: int):
        return DomainElement(*args)


class IDomain(ABC):

    @abstractmethod
    def getCardinality(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def getComponent(self, index: int):
        raise NotImplementedError

    @abstractmethod
    def getNumberOfComponents(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def indexOfElement(self, DomainElement) -> int:
        raise NotImplementedError

    @abstractmethod
    def elementForIndex(self, n: int):
        raise NotImplementedError


class Domain(IDomain):

    def __init__(self):
        pass

    @staticmethod
    def intRange(min: int, max: int):
        return SimpleDomain(min, max)

    @staticmethod
    def combine(domain1, domain2):
        return CompositeDomain(domain1, domain2)

    def indexOfElement(self, DomainElement):
        pass

    def elementForIndex(self, index: int):
        pass


class SimpleDomain(Domain):

    def __init__(self, first: int, last: int):
        super().__init__()
        self.first = first
        self.last = last

    def __str__(self):
        return "\n".join(["Element domene: " + str(i) for i in range(self.first, self.last)])

    def getCardinality(self) -> int:
        return self.last - self.first

    def getComponent(self, index: int):
        return self

    def getNumberOfComponents(self) -> int:
        return 1

    def iterator(self):
        pass

    def getFirst(self) -> int:
        return self.first

    def getLast(self) -> int:
        return self.last

    def elementForIndex(self, index: int):
        return DomainElement(self.first + index)

    def indexOfElement(self, domainElement):
        if domainElement.getNumberOfComponents() > 1:
            return -1
        else:
            elements = [DomainElement(i) for i in range(self.first, self.last)]
            for i, element in enumerate(elements):
                if element.equals(domainElement):
                    return i
            return -1


class CompositeDomain(Domain):

    def __init__(self, *args):
        super().__init__()
        self.SimpleDomains = list(args)

    def __str__(self):
        elements = [[i for i in range(simpleDomain.getFirst(), simpleDomain.getLast())] for simpleDomain in self.SimpleDomains]
        cartesianProducts = list(product(*elements))
        return "\n".join(["Element domene: " + str(cartesianProduct) for cartesianProduct in cartesianProducts])

    def iterator(self):
        pass

    def getCardinality(self):
        return prod([simpleDomain.getCardinality() for simpleDomain in self.SimpleDomains])

    def getComponent(self, index: int):
        return self.SimpleDomains[index]

    def getNumberOfComponents(self) -> int:
        return len(self.SimpleDomains)

    def elementForIndex(self, index: int):
        elements = [[i for i in range(simpleDomain.getFirst(), simpleDomain.getLast())] for simpleDomain in
                    self.SimpleDomains]
        return DomainElement(*list(product(*elements))[index])

    def indexOfElement(self, domainElementTest):
        if domainElementTest.getNumberOfComponents() != self.getNumberOfComponents():
            return -1
        else:
            elements = list(product(*[[i for i in range(simpleDomain.getFirst(), simpleDomain.getLast())] for simpleDomain in
                        self.SimpleDomains]))
            domainElements = [DomainElement(*element) for element in elements]
            for i, domainElement in enumerate(domainElements):
                if domainElement.equals(domainElementTest):
                    return i
            return -1


class IFuzzySet(ABC):

    @abstractmethod
    def getDomain(self):
        pass

    @abstractmethod
    def getValueAt(self, domainElement) -> float:
        pass


class IInitUnaryFunction(ABC):

    def valueAt(self, e: int) -> float:
        pass


class CalculatedFuzzySet(IDomain, IFuzzySet, IInitUnaryFunction):

    def __init__(self, IDomain, IInitUnaryFunction):
        self.IDomain = IDomain
        self.IIinitUnaryFunction = IInitUnaryFunction

    def getDomain(self):
        pass

    def getValueAt(self, DomainElement):
        pass

    def valueAt(self, e: int) -> float:
        pass


class MutableFuzzySet(IFuzzySet):

    memberships = []

    def __init__(self, IDomain):
        self.IDomain = IDomain
        self.memberships = [0.0 for i in range(IDomain.getCardinality())]

    def __str__(self):
        return "\n".join([f"d({i})={round(value, 6)}" for i, value in zip(range(self.IDomain.getFirst(),
                                                                                self.IDomain.getLast()),self.memberships)])

    def set(self, domainElement, double):
        if (index := self.IDomain.indexOfElement(domainElement)) != -1:
            self.memberships[index] = double

    def getDomain(self):
        return self.IDomain

    def getValueAt(self, domainElement) -> float:
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
    d1 = Domain.intRange(0, 5)
    print(d1)
    print(f"Kardinalitet domene je: {d1.getCardinality()}")

    d2 = Domain.intRange(0, 3)
    print(d2)
    print(f"Kardinalitet domene je: {d2.getCardinality()}")

    d3 = Domain.combine(d1, d2)
    print(d3)
    print(f"Kardinalitet domene je: {d3.getCardinality()}")

    print(d3.elementForIndex(0))
    print(d3.elementForIndex(5))
    print(d3.elementForIndex(14))
    print(d3.indexOfElement(DomainElement.of(4,1)))

    d = Domain.intRange(0, 11)
    set1 = MutableFuzzySet(d)
    set1.set(DomainElement.of(0), 1.0)
    set1.set(DomainElement.of(1), 0.8)
    set1.set(DomainElement.of(2), 0.6)
    set1.set(DomainElement.of(3), 0.4)
    set1.set(DomainElement.of(4), 0.2)
    print("Set1", set1)

    d2_F = Domain.intRange(-5, 6)
    set2 = CalculatedFuzzySet(d2, StandardFuzzySets.lambdaFunction(
                d2.indexOfElement(DomainElement.of(-4)),
                d2.indexOfElement(DomainElement.of( 0)),
                d2.indexOfElement(DomainElement.of( 4))
                ))

